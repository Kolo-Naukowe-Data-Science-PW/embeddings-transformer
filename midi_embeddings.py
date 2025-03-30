import hydra
import torch
from omegaconf import OmegaConf, DictConfig

from utility.train import train
from utility.eval import evaluate_model


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    # Determine mode based on config
    if hasattr(cfg, "eval") and cfg.eval.enabled:
        evaluate_model(cfg, device)
    else:
        train(cfg, device)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
