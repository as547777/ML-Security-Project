"use client"
import React from "react"
import { useData } from "@/context/DataContext"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Info } from "lucide-react"

const HyperparameterInputs = () => {
  const { learningRate, setLearningRate, epochs, setEpochs } = useData()

  return (
    <TooltipProvider delayDuration={150}>
      <div className="flex flex-wrap gap-4">
        {/* Learning Rate */}
        <div className="param-container">
          <div className="flex w-full justify-between gap-1 mb-1">
            <label className="param-label">Learning Rate</label>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info
                  size={14}
                  className="text-zinc-400 hover:text-blue-500 transition-colors"
                />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs text-sm text-zinc-200 pb-2">
                Controls how much to adjust the model’s weights after each update.
                Smaller values make training slower but more stable.
              </TooltipContent>
            </Tooltip>
          </div>

          <input
            type="number"
            step="0.001"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            className="param-input"
          />
        </div>

        {/* Epochs */}
        <div className="param-container">
          <div className="flex w-full justify-between gap-1 mb-1">
            <label className="param-label">Epochs</label>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info
                  size={14}
                  className="text-zinc-400 hover:text-blue-500 cursor-pointer transition-colors"
                />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs text-sm text-zinc-200 pb-2">
                The number of times the model sees the entire training dataset.
              </TooltipContent>
            </Tooltip>
          </div>

          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value))}
            className="param-input"
          />
        </div>
      </div>
    </TooltipProvider>
  )
}

export default HyperparameterInputs
