use futures::Future;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trace {
    pub label: String,
    pub latency: Option<u64>,
    pub metadata: Option<String>,
    pub inner_traces: Option<Vec<Trace>>,
}

impl std::fmt::Display for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn recursive(trace: Trace, depth: u16) -> String {
            let mut result = "".to_string();

            let indent = "    ".repeat(depth as usize);
            result.push_str(format!("{}|start {}\n", indent, trace.label).as_str());

            if let Some(inners) = trace.inner_traces {
                inners.into_iter().for_each(|inner| {
                    let inner_result = recursive(inner, depth + 1);

                    result.push_str(inner_result.as_str());
                });
            }

            result.push_str(
                format!(
                    "{}|end {}. {}, latency {}\n",
                    indent,
                    trace.label,
                    trace.metadata.unwrap_or_default().replace("\n", ""),
                    trace.latency.unwrap()
                )
                .as_str(),
            );

            result
        }

        write!(f, "{}", recursive(self.clone(), 0))
    }
}

impl Trace {
    pub fn start(label: &str) -> (Self, Instant) {
        (
            Self {
                label: label.to_string(),
                latency: None,
                metadata: None,
                inner_traces: None,
            },
            Instant::now(),
        )
    }

    pub fn end(
        &mut self,
        start: Instant,
        metadata: Option<String>,
        inner_traces: Option<Vec<Trace>>,
    ) {
        self.latency = Some(start.elapsed().as_millis() as u64);
        self.metadata = metadata;
        self.inner_traces = inner_traces;
    }
}

pub async fn instrument<F, T, E>(
    f: impl FnOnce() -> F,
    label: &str,
    meta_data: impl FnOnce(&T) -> Option<String>,
    inner_traces: impl FnOnce(&T) -> Option<Vec<Trace>>,
) -> Result<(T, Trace), E>
where
    F: Future<Output = Result<T, E>>,
{
    let (mut trace, start) = Trace::start(label);
    let res = f().await?;
    trace.end(start, meta_data(&res), inner_traces(&res));

    Ok((res, trace))
}
