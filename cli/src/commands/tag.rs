use anyhow::{bail, Result};
use std::path::Path;

use crate::cli::TagArgs;
use crate::data::experiment::discover_experiments;
use crate::data::meta;

pub fn run(args: TagArgs, base_dir: &Path) -> Result<()> {
    let experiments = discover_experiments(base_dir);
    let exp = experiments
        .iter()
        .find(|e| e.name == args.experiment)
        .ok_or_else(|| anyhow::anyhow!("experiment '{}' not found", args.experiment))?;

    let mut tags = meta::read_tags(&exp.dir);

    if args.list {
        if tags.is_empty() {
            println!("No tags for '{}'", args.experiment);
        } else {
            println!("Tags for '{}':", args.experiment);
            for tag in &tags {
                println!("  {}", tag);
            }
        }
        return Ok(());
    }

    if let Some(ref remove_tag) = args.remove {
        if let Some(pos) = tags.iter().position(|t| t == remove_tag) {
            tags.remove(pos);
            meta::update_tags(&exp.dir, &tags)?;
            println!("Removed tag '{}' from '{}'", remove_tag, args.experiment);
        } else {
            bail!("tag '{}' not found on '{}'", remove_tag, args.experiment);
        }
        return Ok(());
    }

    if let Some(ref add_tag) = args.tag {
        if tags.contains(add_tag) {
            println!("Tag '{}' already exists on '{}'", add_tag, args.experiment);
        } else {
            tags.push(add_tag.clone());
            meta::update_tags(&exp.dir, &tags)?;
            println!("Added tag '{}' to '{}'", add_tag, args.experiment);
        }
        return Ok(());
    }

    // No action specified
    bail!("specify a tag to add, --remove <tag>, or --list");
}
