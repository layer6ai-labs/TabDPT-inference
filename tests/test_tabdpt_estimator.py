from unittest.mock import MagicMock, patch

from tabdpt import estimator


@patch("tabdpt.estimator.hf_hub_download", return_value="my/hf/path")
@patch("tabdpt.estimator.safe_open")
@patch("tabdpt.estimator.torch.cuda.is_available", return_value=False)
@patch("tabdpt.estimator.json.loads", return_value={"env": {}})
@patch("tabdpt.estimator.OmegaConf.create")
@patch("tabdpt.estimator.TabDPTModel.load", return_value=MagicMock(num_features=10, n_out=2))
def test_model_weight_path_set_or_download_from_hf(
    mock_model_load,
    mock_omega_create,
    mock_json_loads,
    mock_cuda_available,
    mock_safe_open,
    mock_hf_hub_download,
):
    mock_file = MagicMock()
    mock_file.metadata.return_value = {"cfg": "{}"}
    mock_file.keys.return_value = ["tensor1"]
    mock_file.get_tensor.return_value = MagicMock()
    mock_safe_open.return_value.__enter__.return_value = mock_file

    # load from local path
    local_path = "/test/path/model.safetensors"
    dpt_estimator = estimator.TabDPTEstimator(mode="cls", model_weight_path=local_path)
    mock_hf_hub_download.assert_not_called()
    assert dpt_estimator.path == local_path

    # load from hf path
    dpt_estimator = estimator.TabDPTEstimator(mode="cls")
    mock_hf_hub_download.assert_called_once_with(
        repo_id="Layer6/TabDPT",
        filename=estimator._MODEL_NAME,
    )
    assert dpt_estimator.path == "my/hf/path"
