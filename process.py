import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from typing import Dict
from pathlib import Path
from statistics import mean

import torch
import SimpleITK
import segmentation_models
from tensorflow.keras.models import load_model

from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from utils import MultiClassAlgorithm, to_input_format, device
from algorithm.model import get_model
from algorithm.preprocess import preprocess
from algorithm.prep_seg import prep_seg


COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")


class StoicAlgorithm(MultiClassAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/"),
            # input_path=Path("./test/images/ct/"),
            # output_path=Path("./test/"),
        )

        # load model
        kwargs = {"map_location": torch.device(device)}
        self.seg_model = load_model('./algorithm/lung_inf_stoic.hdf5', compile=False)
        
        self.covid_model = get_model()
        self.covid_model.load_state_dict(torch.load('./algorithm/covid_model.pth', **kwargs))
        self.covid_model.eval()
        
        self.fold_models = []
        for fold in range(5):
            model = get_model()
            model.load_state_dict(torch.load(f'./algorithm/f{fold}.pth', **kwargs))
            model.eval()
            self.fold_models.append(model)

    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        # pre-processing
        input_image = prep_seg(input_image, self.seg_model)
        input_image = preprocess(input_image)
        input_image = to_input_format(input_image)

        # run model
        with torch.no_grad():
            covid = torch.sigmoid(self.covid_model(input_image)).item()
            if covid < 0.4:
                prob_covid, prob_severe = covid, 0.0
            else:
                severe = mean([torch.sigmoid(model(input_image)).item() for model in self.fold_models])
                prob_covid, prob_severe = covid, severe

        return {
            COVID_OUTPUT_NAME: prob_covid,
            SEVERE_OUTPUT_NAME: prob_severe
        }


if __name__ == "__main__":
    StoicAlgorithm().process()
