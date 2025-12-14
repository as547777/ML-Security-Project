"use client"
import React from "react"
import {useData} from "@/context/DataContext";

const DatasetInfoCard = () => {
  const { dataset } = useData();

  if (!dataset) {
    return <p>Select a dataset to see details.</p>
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight text-blue-700">
          {dataset.name}
        </h2>
        <span className="text-sm bg-blue-100 text-blue-700 px-3 py-1 rounded-full border border-blue-200">
          {dataset.type}
        </span>
      </div>

      {/* Description */}
      <p className="text-zinc-600 leading-relaxed">{dataset.description}</p>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-3 text-sm mt-2">
        <div className="flex flex-col items-center justify-center rounded-xl bg-blue-50/60 p-3 border border-blue-100 shadow-sm">
          <span className="text-xs text-zinc-500 uppercase tracking-wide">
            Train Samples
          </span>
          <span className="text-lg font-semibold text-blue-700">
            {dataset.trainCount.toLocaleString()}
          </span>
        </div>

        <div className="flex flex-col items-center justify-center rounded-xl bg-purple-50/60 p-3 border border-purple-100 shadow-sm">
          <span className="text-xs text-zinc-500 uppercase tracking-wide">
            Test Samples
          </span>
          <span className="text-lg font-semibold text-purple-700">
            {dataset.testCount.toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  )
}

export default DatasetInfoCard
