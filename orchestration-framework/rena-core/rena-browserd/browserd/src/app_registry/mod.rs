use error_stack::ResultExt;
use octocrab::{repos::RepoHandler, Octocrab};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs::{read, read_dir};
use std::future::Future;
use std::path::Path;
use thiserror::Error;

use crate::discovery::AppFile;

pub mod rena;

pub mod proto {
    pub mod common {
        tonic::include_proto!("common");
    }

    pub mod app_registry {
        tonic::include_proto!("app_registry");
    }
}

type Result<T> = error_stack::Result<T, Error>;

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("local registry failed")]
    LocalRegistry,
    #[error("github registry failed")]
    GithubRegistry,
    #[error("rena registry failed")]
    RenaRegistry,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum Config {
    #[serde(rename = "rena")]
    Rena { url: String },
}

pub trait RegistryClient {
    fn pull(&self, path: &str) -> impl Future<Output = Result<Option<Vec<AppFile>>>>;
}

#[derive(Clone)]
pub struct LocalRegistry;

impl RegistryClient for LocalRegistry {
    async fn pull(&self, path: &str) -> Result<Option<Vec<AppFile>>> {
        let re = Regex::new(r"^file://(.+)").expect("compile regex failed");

        if let Some(capture) = re.captures(path) {
            let content_path = capture
                .get(1)
                .ok_or(Error::LocalRegistry)
                .attach_printable("invalid local path")?
                .as_str();

            return read_local_content(Path::new(content_path), Path::new("."), &mut Vec::new());
        }

        Err(Error::LocalRegistry.into())
    }
}

#[derive(Clone)]
pub struct GithubRegistry;

impl RegistryClient for GithubRegistry {
    async fn pull(&self, path: &str) -> Result<Option<Vec<AppFile>>> {
        let re = Regex::new(r"^(?:https://)?github\.com/([^/]+)/([^/]+)(?:/tree/)?([^/]+)?/?(.+)?")
            .expect("compile regex failed");

        if let Some(captures) = re.captures(path) {
            let owner = captures
                .get(1)
                .ok_or(Error::GithubRegistry)
                .attach_printable("gitub org/owner not found")?
                .as_str();
            let repo = captures
                .get(2)
                .ok_or(Error::GithubRegistry)
                .attach_printable("gitub repo not found")?
                .as_str();

            let (branch, content_path) = match captures.get(3) {
                Some(branch_match) => match captures.get(4) {
                    Some(content_path_match) => {
                        (branch_match.as_str(), content_path_match.as_str())
                    }
                    None => (branch_match.as_str(), "."),
                },
                None => ("main", "."),
            };

            let octocrab = Octocrab::builder()
                .build()
                .change_context(Error::GithubRegistry)?;
            let repo = octocrab.repos(owner, repo);

            return pull_github_content(&repo, branch, content_path, ".", &mut Vec::new()).await;
        }

        Err(Error::GithubRegistry.into())
    }
}

#[derive(Clone)]
pub struct Registry {
    pub local: LocalRegistry,
    pub github: GithubRegistry,
    pub rena: Option<rena::RenaRegistry>,
}

impl Registry {
    pub async fn new(config: &Option<Config>) -> Result<Self> {
        let rena = match config {
            Some(Config::Rena { url }) => Some(rena::RenaRegistry::new(&url).await?),
            None => None,
        };

        Ok(Self {
            local: LocalRegistry,
            github: GithubRegistry,
            rena,
        })
    }

    pub async fn pull(&self, path: &str) -> Result<Option<Vec<AppFile>>> {
        if path.starts_with("file://") {
            return self.local.pull(path).await;
        }

        if path.contains("github.com") {
            return self.github.pull(path).await;
        }

        self.rena
            .as_ref()
            .ok_or(Error::RenaRegistry)
            .attach_printable("rena registry not configured")?
            .pull(path)
            .await
    }
}

fn read_local_content(
    content_path: &Path,
    relative_path: &Path,
    results: &mut Vec<AppFile>,
) -> Result<Option<Vec<AppFile>>> {
    if !content_path.exists() {
        return Err(Error::LocalRegistry.into());
    }

    match content_path.is_dir() {
        true => {
            for entry in read_dir(content_path).change_context(Error::LocalRegistry)? {
                let entry = entry.change_context(Error::LocalRegistry)?;

                if entry.path().is_dir() {
                    read_local_content(
                        &entry.path(),
                        &relative_path.join(entry.file_name()),
                        results,
                    )?;
                } else if entry.path().is_file() {
                    results.push(AppFile {
                        relative_path: relative_path
                            .join(entry.file_name())
                            .to_str()
                            .ok_or(Error::LocalRegistry)?
                            .to_string(),
                        content: read(entry.path()).change_context(Error::LocalRegistry)?,
                    });
                }
            }
        }
        false => results.push(AppFile {
            relative_path: relative_path
                .join(content_path.file_name().ok_or(Error::LocalRegistry)?)
                .to_str()
                .ok_or(Error::LocalRegistry)?
                .to_string(),
            content: read(content_path).change_context(Error::LocalRegistry)?,
        }),
    }

    Ok(Some(results.clone()))
}

async fn pull_github_content(
    repo: &RepoHandler<'_>,
    branch: &str,
    content_path: &str,
    relative_path: &str,
    results: &mut Vec<AppFile>,
) -> Result<Option<Vec<AppFile>>> {
    let contents = repo
        .get_content()
        .path(content_path)
        .r#ref(branch)
        .send()
        .await
        .change_context(Error::GithubRegistry)?
        .items;

    for content in contents {
        match content.r#type.as_str() {
            "dir" => {
                Box::pin(pull_github_content(
                    repo,
                    branch,
                    &content.path,
                    &format!("{}/{}", relative_path, content.name),
                    results,
                ))
                .await?;
            }
            "file" => results.push(AppFile {
                relative_path: format!("{}/{}", relative_path, content.name),
                content: reqwest::get(
                    content
                        .download_url
                        .ok_or(Error::GithubRegistry)
                        .attach_printable("download_url not found")?,
                )
                .await
                .change_context(Error::GithubRegistry)?
                .bytes()
                .await
                .change_context(Error::GithubRegistry)?
                .to_vec(),
            }),
            _ => {}
        }
    }

    Ok(Some(results.clone()))
}
