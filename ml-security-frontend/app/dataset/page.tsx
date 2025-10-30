"use client"

import MainContainer from "@/components/main-container"
import { StepNavigation } from "@/components/step-navigation"
import DatasetSelect from "@/components/dataset-select"
import HyperparameterInputs from "@/components/hyperparameter-inputs"
import DatasetInfoCard from "@/components/dataset-info-card"

// TODO - ovo zamjenit sa fetchom na bek
const datasetList = [
  {
    name: "MNIST",
    description:
      "The MNIST dataset contains 70,000 images of handwritten digits (0–9). Each image is 28×28 grayscale.",
    type: "Image",
    trainCount: 60000,
    testCount: 10000,
  },
  {
    name: "CIFAR-10",
    description:
      "CIFAR-10 consists of 60,000 32×32 color images in 10 classes, with 6,000 images per class.",
    type: "Image",
    trainCount: 50000,
    testCount: 10000,
  },
  {
    name: "Custom Dataset",
    description:
      "A user-provided dataset. Details and statistics will depend on your upload or configuration.",
    type: "",
    trainCount: 0,
    testCount: 0,
  },
]

export default function DatasetPage() {
  return (
    <MainContainer>
      <h1 className="text-4xl md:text-5xl header-text mb-7 p-1">
        Datasets & Hyperparameters
      </h1>

      <div className="flex flex-col lg:flex-row items-start gap-6">
        {/* Left Card */}
        <div className="card-main w-full max-w-xl flex-1 flex flex-col justify-between">
          <div>
            <p className="text-zinc-600 mb-4">
              Here you can select the dataset you’ll use and set the basic network hyperparameters.
            </p>

            <div className="flex flex-col gap-4 text-zinc-700">
              <DatasetSelect datasets={datasetList} />
              <HyperparameterInputs />
            </div>
          </div>

          {/* Docked Navigation */}
          <div className="mt-8">
            <StepNavigation next="/attack-defense" />
          </div>
        </div>

        {/* Right Card */}
        <div className="card-main w-md min-h-md">
          <DatasetInfoCard />
        </div>
      </div>
    </MainContainer>
  )
}
