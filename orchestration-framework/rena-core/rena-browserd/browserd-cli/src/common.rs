use config::{Config, ConfigError, File};
use serde::Deserialize;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("config failed")]
    Config,
    #[error("browserd failed")]
    Browserd,
    #[error("REPL failed")]
    Repl,
    #[error("query failed")]
    Query,
    #[error("eval failed")]
    Eval,
}

pub type Result<T> = error_stack::Result<T, Error>;

pub fn init_config<'de, T: Deserialize<'de>>(
    config_path: &str,
) -> error_stack::Result<T, ConfigError> {
    Config::builder()
        .add_source(File::with_name(config_path))
        .build()?
        .try_deserialize::<T>()
        .map_err(error_stack::Report::from)
}
