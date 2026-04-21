use super::super::http::respond_json;
use super::super::status::ModelTargetPayload;
use super::super::MeshApi;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct ModelTargetsResponse {
    model_targets: Vec<ModelTargetPayload>,
}

pub(super) async fn handle(
    stream: &mut tokio::net::TcpStream,
    state: &MeshApi,
) -> anyhow::Result<()> {
    respond_json(
        stream,
        200,
        &ModelTargetsResponse {
            model_targets: state.model_targets().await,
        },
    )
    .await
}
