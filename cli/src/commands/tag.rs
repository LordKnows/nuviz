use anyhow::{bail, Result};
use std::path::Path;

use crate::cli::TagArgs;
use crate::commands::common::find_experiment;
use crate::data::meta;

pub fn run(args: TagArgs, base_dir: &Path) -> Result<()> {
    let exp = find_experiment(base_dir, &args.experiment, args.project.as_deref())?;
    let label = exp.qualified_name();
    let mut tags = meta::read_tags(&exp.dir);

    if args.list {
        if tags.is_empty() {
            println!("No tags for '{label}'");
        } else {
            println!("Tags for '{label}':");
            for tag in &tags {
                println!("  {tag}");
            }
        }
        return Ok(());
    }

    if let Some(ref remove_tag) = args.remove {
        let Some(pos) = tags.iter().position(|t| t == remove_tag) else {
            bail!("tag '{remove_tag}' not found on '{label}'");
        };
        tags.remove(pos);
        meta::update_tags(&exp.dir, &tags)?;
        println!("Removed tag '{remove_tag}' from '{label}'");
        return Ok(());
    }

    if let Some(ref add_tag) = args.tag {
        if tags.contains(add_tag) {
            println!("Tag '{add_tag}' already exists on '{label}'");
        } else {
            tags.push(add_tag.clone());
            meta::update_tags(&exp.dir, &tags)?;
            println!("Added tag '{add_tag}' to '{label}'");
        }
        return Ok(());
    }

    bail!("specify a tag to add, --remove <tag>, or --list");
}
