use error_stack::ResultExt;
use tonic::transport::Channel;

use super::proto::app_registry::{app_registry_client, PullAppRequest};
use super::proto::common;
use super::{Error, RegistryClient, Result};

use crate::discovery;

impl From<common::AppFile> for discovery::AppFile {
    fn from(app_file: common::AppFile) -> Self {
        Self {
            relative_path: app_file.relative_path,
            content: app_file.content,
        }
    }
}

impl From<common::AppFileList> for Vec<discovery::AppFile> {
    fn from(value: common::AppFileList) -> Self {
        value.files.into_iter().map(Into::into).collect()
    }
}

#[derive(Clone)]
pub struct RenaRegistry {
    pub url: String,
    pub inner: app_registry_client::AppRegistryClient<Channel>,
}

impl RenaRegistry {
    pub async fn new(url: &str) -> Result<Self> {
        let channel = Channel::from_shared(url.to_string())
            .change_context(Error::RenaRegistry)?
            .connect()
            .await
            .change_context(Error::RenaRegistry)?;

        let mut client = app_registry_client::AppRegistryClient::new(channel);
        client = client
            .max_decoding_message_size(256 * 1024 * 1024) // 256MB
            .max_encoding_message_size(256 * 1024 * 1024); // 256MB

        Ok(Self {
            url: url.to_string(),
            inner: client,
        })
    }
}

impl RegistryClient for RenaRegistry {
    async fn pull(&self, path: &str) -> Result<Option<Vec<discovery::AppFile>>> {
        let prefix = if self.url.ends_with('/') {
            self.url.clone()
        } else {
            format!("{}/", self.url)
        };

        let req = PullAppRequest {
            app_name: path
                .strip_prefix(&prefix)
                .ok_or(Error::RenaRegistry)?
                .to_string(),
        };

        match self
            .inner
            .clone()
            .pull(req)
            .await
            .change_context(Error::RenaRegistry)?
            .into_inner()
            .app_files
        {
            Some(app_files) => Ok(Some(app_files.into())),
            None => Ok(None),
        }
    }
}
