"use client"

import MainContainer from "@/components/main-container"
import { StepNavigation } from "@/components/step-navigation"
import { useData } from "@/context/DataContext"
import {useState} from "react";

export default function OverviewPage() {
  const { dataset, momentum, learningRate, epochs, attack, attackParams } = useData()
  const [isRunning, setIsRunning] = useState(false)

  // TODO - provjerit zasto proxy ne ceka response neg faila
  // TODO - prebacit u konkretnu next.js api datoteku
  const execute = async () => {
    setIsRunning(true);
    try {
      const attackParamsValues = attackParams
        ? Object.entries(attackParams).reduce((acc, [key, param]) => {
          acc[key] = param.value;
          return acc;
        }, {} as Record<string, number | string>)
        : {};

      const response = await fetch('http://localhost:3000/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          "dataset": dataset?.name,
          "learning_rate": learningRate,
          "epochs": epochs,
          "momentum": momentum,
          "attack": attack?.name,
          "model": "ImageModel",
          ...attackParamsValues
        })
      })
      if (!response.ok) {
        throw new Error('Network response was not ok')
      }
      const data = await response.json()
      alert(JSON.stringify(data))
    }
    catch (error) {
      alert(error)
    }
    finally {
      setIsRunning(false);
    }
  }

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

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
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
              <h3 className="text-sm font-medium text-zinc-500">Momentum</h3>
              <p className="text-lg font-semibold text-zinc-800 mt-1">{momentum}</p>
            </div>

            <div className={"block h-4"}></div>
            <div className={"block"}></div>

            <div className="rounded-2xl bg-white/80 backdrop-blur-md p-4 shadow-sm border border-white/30">
              <h3 className="text-sm font-medium text-zinc-500">Attack</h3>
              <p className="text-lg font-semibold text-zinc-800 mt-1">{attack?.name || "Not selected"}</p>
            </div>
          </div>
        </div>

        <StepNavigation
          prev="/attack-defense"
          isFinal={true}
          isRunning={isRunning}
          onRun={execute}
        />
      </div>
    </MainContainer>
  )
}
