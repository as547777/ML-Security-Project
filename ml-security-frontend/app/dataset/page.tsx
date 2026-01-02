import MainContainer from "@/components/main-container"
import { StepNavigation } from "@/components/step-navigation"
import DatasetSelect from "@/components/dataset/dataset-select"
import HyperparameterInputs from "@/components/hyperparameter-inputs"
import DatasetInfoCard from "@/components/dataset/dataset-info-card"
import {DatasetInfo} from "@/types";
import ModelSelect from "@/components/model/model-select";
import Card from "@/components/card";

export default async function DatasetPage() {
  const dataset_data = await fetch('http://localhost:5000/datasets')
  // TODO - add optimizer and loss function to backend
  // const optimizer_data = await fetch('http://localhost:3004/optimizers')
  // const loss_function_data = await fetch('http://localhost:3004/loss_functions')

  const datasets = await dataset_data.json() as DatasetInfo[]
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
            <ModelSelect />
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
