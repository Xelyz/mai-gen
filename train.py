import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pytorch_lightning.cli import LightningCLI
from models.seq2seq import ChartGenerator
from data.dataset import MaiGenDataModule

def cli_main():
    cli = LightningCLI(
        model_class=ChartGenerator,
        datamodule_class=MaiGenDataModule,
        save_config_kwargs={"overwrite": True}
    )

if __name__ == '__main__':
    cli_main()
