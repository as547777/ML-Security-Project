"use client"

import MainContainer from "@/components/main-container"
import { StepNavigation } from "@/components/step-navigation"
import DatasetSelect from "@/components/dataset-select"
import HyperparameterInputs from "@/components/hyperparameter-inputs"
import DatasetInfoCard from "@/components/dataset-info-card"

export default function DatasetPage() {
  return (
    <MainContainer>
      <h1 className="text-4xl md:text-5xl header-text mb-7 p-1">
        Datasets & Hyperparameters
      </h1>

      <div className="flex flex-col lg:flex-row items-start gap-6">
        {/* Left Card */}
        <div className="card-main w-full max-w-2xl flex-1 flex flex-col justify-between">
          <div>
            <p className="text-zinc-600 mb-4">
              Here you can select the dataset you’ll use and set the basic network hyperparameters.
            </p>

            <div className="flex flex-col gap-4 text-zinc-700">
              <DatasetSelect />
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
