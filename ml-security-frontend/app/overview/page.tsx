"use client"

import MainContainer from "@/components/main-container"
import { StepNavigation } from "@/components/step-navigation"
import { useData } from "@/context/DataContext"

export default function OverviewPage() {
  const { dataset, learningRate, epochs, attack, defense } = useData()

  return (
    // TODO - ovo promjenit skroz
    <MainContainer>
      <h1 className="text-4xl md:text-5xl header-text mb-7 p-1">
        Overview
      </h1>

      <div className="card-main w-full max-w-2xl text-zinc-700">
        <div className="space-y-6">
          <p className="text-zinc-600">
            Here’s a summary of your selected configuration:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="rounded-2xl bg-white/80 backdrop-blur-md p-4 shadow-sm border border-white/30">
              <h3 className="text-sm font-medium text-zinc-500">Dataset</h3>
              <p className="text-lg font-semibold text-zinc-800 mt-1">{dataset?.name || "Not selected"}</p>
            </div>

            <div className="rounded-2xl bg-white/80 backdrop-blur-md p-4 shadow-sm border border-white/30">
              <h3 className="text-sm font-medium text-zinc-500">Learning Rate</h3>
              <p className="text-lg font-semibold text-zinc-800 mt-1">{learningRate}</p>
            </div>

            <div className="rounded-2xl bg-white/80 backdrop-blur-md p-4 shadow-sm border border-white/30">
              <h3 className="text-sm font-medium text-zinc-500">Epochs</h3>
              <p className="text-lg font-semibold text-zinc-800 mt-1">{epochs}</p>
            </div>

            <div className="rounded-2xl bg-white/80 backdrop-blur-md p-4 shadow-sm border border-white/30">
              <h3 className="text-sm font-medium text-zinc-500">Attack</h3>
              <p className="text-lg font-semibold text-zinc-800 mt-1">{attack || "Not selected"}</p>
            </div>

            <div className="rounded-2xl bg-white/80 backdrop-blur-md p-4 shadow-sm border border-white/30">
              <h3 className="text-sm font-medium text-zinc-500">Defense</h3>
              <p className="text-lg font-semibold text-zinc-800 mt-1">{defense || "Not selected"}</p>
            </div>
          </div>
        </div>

        <StepNavigation
          prev="/attack-defense"
          isFinal={true}
          onRun={() => console.log("Run started")}
        />
      </div>
    </MainContainer>
  )
}
