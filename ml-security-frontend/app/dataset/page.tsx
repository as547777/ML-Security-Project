import MainContainer from "@/components/main-container"
import { StepNavigation } from "@/components/step-navigation"
import DatasetSelect from "@/components/dataset/dataset-select"
import HyperparameterInputs from "@/components/hyperparameter-inputs"
import {DatasetInfo, ModelInfo} from "@/types";
import ModelSelect from "@/components/model/model-select";
import Card from "@/components/card";

export default async function DatasetPage() {
  const API_URL = process.env.NEXT_PUBLIC_API_URL_SERVER || "http://localhost:5000";

  const dataset_data = await fetch(API_URL + '/datasets')
  const models_temp_data = await fetch(API_URL +  '/models?type=image')
  // TODO - add optimizer and loss function to backend
  // const optimizer_data = await fetch('http://localhost:3004/optimizers')
  // const loss_function_data = await fetch('http://localhost:3004/loss_functions')

  const datasets = await dataset_data.json() as DatasetInfo[]
  const models_temp = await models_temp_data.json() as ModelInfo[]
  // const optimizers = await optimizer_data.json()
  // const lossFunctions = await loss_function_data.json()

  return (
    <MainContainer>
      <h1 className="text-4xl md:text-5xl header-text mb-7 p-1">
        Datasets & Models
      </h1>

      <Card>
        <div>
          <p className="text-zinc-600 mb-4">
            Here you can select the dataset you’ll use and set the basic network hyperparameters.
          </p>

          <div className="flex flex-col gap-4 text-zinc-700">
            <DatasetSelect datasets={datasets} />
            <ModelSelect modelFamilies={models_temp} />
            <HyperparameterInputs />
            {/*<HyperparameterInputs optimizers={optimizers} lossFunctions={lossFunctions} />*/}
          </div>
        </div>

        <div className="mt-8">
          <StepNavigation next="/attack-defense" />
        </div>
      </Card>
    </MainContainer>
  )
}
