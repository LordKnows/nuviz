//! Resolving what the user typed to a single experiment.
//!
//! Accepted forms, in order of precedence:
//!
//! 1. `project/name`
//! 2. a directory name or `meta.name` (optionally narrowed by `--project`)
//! 3. a `run_id` prefix (at least 4 characters)
//! 4. a path to an experiment directory, inside or outside the base directory
//!
//! An ambiguous query is an error that lists the candidates, rather than
//! silently picking whichever experiment happened to be discovered first.

use std::path::Path;

use anyhow::{anyhow, bail, Result};

use crate::data::experiment::{is_experiment_dir, Experiment};

pub fn resolve(
    experiments: &[Experiment],
    query: &str,
    project: Option<&str>,
) -> Result<Experiment> {
    let in_project = |e: &&Experiment| project.is_none_or(|p| e.project.as_deref() == Some(p));

    // 1. project/name
    if let Some((proj, name)) = query.split_once('/') {
        let matches: Vec<&Experiment> = experiments
            .iter()
            .filter(|e| {
                e.project.as_deref() == Some(proj) && (e.dir_name == name || e.name == name)
            })
            .collect();
        if let Some(found) = pick(&matches, name, query)? {
            return Ok(found.clone());
        }
    }

    // 2. directory name or display name
    let matches: Vec<&Experiment> = experiments
        .iter()
        .filter(in_project)
        .filter(|e| e.dir_name == query || e.name == query)
        .collect();
    if let Some(found) = pick(&matches, query, query)? {
        return Ok(found.clone());
    }

    // 3. run_id prefix
    if query.len() >= 4 {
        let matches: Vec<&Experiment> = experiments
            .iter()
            .filter(in_project)
            .filter(|e| {
                e.meta
                    .run_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with(query))
            })
            .collect();
        match matches.as_slice() {
            [one] => return Ok((*one).clone()),
            [] => {}
            many => bail!(ambiguous(query, many)),
        }
    }

    // 4. filesystem path
    let path = Path::new(query);
    if path.is_dir() && is_experiment_dir(path) {
        return Ok(Experiment::load(path, None));
    }

    Err(not_found(experiments, query, project))
}

/// Resolve several queries, failing on the first that cannot be resolved.
pub fn resolve_many(
    experiments: &[Experiment],
    queries: &[String],
    project: Option<&str>,
) -> Result<Vec<Experiment>> {
    queries
        .iter()
        .map(|q| resolve(experiments, q, project))
        .collect()
}

/// Choose among name matches: a unique match wins, then a unique exact
/// directory-name match; otherwise the query is ambiguous.
fn pick<'a>(
    matches: &[&'a Experiment],
    dir_name: &str,
    query: &str,
) -> Result<Option<&'a Experiment>> {
    match matches {
        [] => Ok(None),
        [one] => Ok(Some(one)),
        many => {
            let exact: Vec<&&Experiment> = many.iter().filter(|e| e.dir_name == dir_name).collect();
            match exact.as_slice() {
                [one] => Ok(Some(one)),
                _ => bail!(ambiguous(query, many)),
            }
        }
    }
}

fn ambiguous(query: &str, candidates: &[&Experiment]) -> String {
    let mut list: Vec<String> = candidates.iter().map(|e| e.qualified_name()).collect();
    list.sort();
    format!(
        "'{query}' is ambiguous; it matches {} experiments:\n  {}\nUse project/name or --project to pick one.",
        list.len(),
        list.join("\n  ")
    )
}

fn not_found(experiments: &[Experiment], query: &str, project: Option<&str>) -> anyhow::Error {
    let needle = query.to_lowercase();
    let mut suggestions: Vec<String> = experiments
        .iter()
        .filter(|e| {
            let name = e.dir_name.to_lowercase();
            name.contains(&needle) || needle.contains(&name) || edit_distance(&name, &needle) <= 2
        })
        .map(|e| e.qualified_name())
        .collect();
    suggestions.sort();
    suggestions.truncate(5);

    let scope = project
        .map(|p| format!(" in project '{p}'"))
        .unwrap_or_default();
    if suggestions.is_empty() {
        anyhow!("experiment '{query}' not found{scope} (run `nuviz ls` to list experiments)")
    } else {
        anyhow!(
            "experiment '{query}' not found{scope}. Did you mean:\n  {}",
            suggestions.join("\n  ")
        )
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur.push((prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::experiment::discover_experiments;
    use crate::data::experiment::test_support::create_experiment;

    fn setup() -> (tempfile::TempDir, Vec<Experiment>) {
        let dir = tempfile::tempdir().unwrap();
        create_experiment(
            dir.path(),
            Some("alpha"),
            "baseline",
            2,
            Some(r#"{"run_id":"aaaa11112222"}"#),
        );
        create_experiment(
            dir.path(),
            Some("beta"),
            "baseline",
            2,
            Some(r#"{"run_id":"bbbb33334444"}"#),
        );
        create_experiment(
            dir.path(),
            Some("beta"),
            "wide",
            2,
            Some(r#"{"run_id":"bbbb55556666"}"#),
        );
        // Legacy pair: re-running a script used to leave two runs with one meta.name.
        create_experiment(dir.path(), None, "exp", 2, Some(r#"{"name":"exp"}"#));
        create_experiment(dir.path(), None, "exp_1", 2, Some(r#"{"name":"exp"}"#));
        let exps = discover_experiments(dir.path());
        (dir, exps)
    }

    #[test]
    fn test_unique_name() {
        let (_d, exps) = setup();
        assert_eq!(
            resolve(&exps, "wide", None).unwrap().qualified_name(),
            "beta/wide"
        );
    }

    #[test]
    fn test_ambiguous_across_projects_is_an_error_listing_candidates() {
        let (_d, exps) = setup();
        let err = resolve(&exps, "baseline", None).unwrap_err().to_string();
        assert!(err.contains("ambiguous"), "{err}");
        assert!(
            err.contains("alpha/baseline") && err.contains("beta/baseline"),
            "{err}"
        );
    }

    #[test]
    fn test_project_qualifier_and_flag_disambiguate() {
        let (_d, exps) = setup();
        assert_eq!(
            resolve(&exps, "alpha/baseline", None)
                .unwrap()
                .project
                .as_deref(),
            Some("alpha")
        );
        assert_eq!(
            resolve(&exps, "baseline", Some("beta"))
                .unwrap()
                .project
                .as_deref(),
            Some("beta")
        );
    }

    #[test]
    fn test_exact_directory_name_beats_shared_display_name() {
        let (_d, exps) = setup();
        assert_eq!(resolve(&exps, "exp", None).unwrap().dir_name, "exp");
        assert_eq!(resolve(&exps, "exp_1", None).unwrap().dir_name, "exp_1");
    }

    #[test]
    fn test_run_id_prefix() {
        let (_d, exps) = setup();
        assert_eq!(
            resolve(&exps, "aaaa", None).unwrap().qualified_name(),
            "alpha/baseline"
        );
        assert!(resolve(&exps, "bbbb", None)
            .unwrap_err()
            .to_string()
            .contains("ambiguous"));
        assert_eq!(resolve(&exps, "bbbb5", None).unwrap().dir_name, "wide");
    }

    #[test]
    fn test_filesystem_path() {
        let (d, exps) = setup();
        let path = d.path().join("alpha").join("baseline");
        let found = resolve(&exps, path.to_str().unwrap(), None).unwrap();
        assert_eq!(found.dir, path);
    }

    #[test]
    fn test_not_found_suggests_close_names() {
        let (_d, exps) = setup();
        let err = resolve(&exps, "wdie", None).unwrap_err().to_string();
        assert!(
            err.contains("not found") && err.contains("beta/wide"),
            "{err}"
        );
        let err = resolve(&exps, "zzzzzzzz", None).unwrap_err().to_string();
        assert!(err.contains("nuviz ls"), "{err}");
    }

    #[test]
    fn test_resolve_many_fails_on_first_missing() {
        let (_d, exps) = setup();
        let ok = resolve_many(&exps, &["wide".into(), "exp_1".into()], None).unwrap();
        assert_eq!(ok.len(), 2);
        assert!(resolve_many(&exps, &["wide".into(), "nope-nope".into()], None).is_err());
    }
}
