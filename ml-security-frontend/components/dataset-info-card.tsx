"use client"
import React, { useEffect, useState } from "react"
import { useData } from "@/context/DataContext"
import Image from "next/image";

interface DatasetInfo {
  name: string
  description: string
  type: string
  trainCount: number
  testCount: number
  preview: string
}

const sampleData: Record<string, DatasetInfo> = {
  MNIST: {
    name: "MNIST",
    description:
      "The MNIST dataset contains 70,000 images of handwritten digits (0–9). Each image is 28×28 grayscale.",
    type: "Image (Grayscale)",
    trainCount: 60000,
    testCount: 10000,
    preview:
      "https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png",
  },
  "CIFAR-10": {
    name: "CIFAR-10",
    description:
      "CIFAR-10 consists of 60,000 32×32 color images in 10 classes, with 6,000 images per class.",
    type: "Image (RGB)",
    trainCount: 50000,
    testCount: 10000,
    preview:
      "https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog4.png",
  },
  "Custom Dataset": {
    name: "Custom Dataset",
    description:
      "A user-provided dataset. Details and statistics will depend on your upload or configuration.",
    type: "Varies",
    trainCount: 0,
    testCount: 0,
    preview:
      "https://cdn-icons-png.flaticon.com/512/3844/3844724.png",
  },
}

const DatasetInfoCard = () => {
  const { dataset } = useData()
  const [info, setInfo] = useState<DatasetInfo | null>(null)

  useEffect(() => {
    let active = true

    async function loadInfo() {
      if (dataset && sampleData[dataset]) {
        // simulate async fetch
        await new Promise((r) => setTimeout(r, 200))
        if (active) setInfo(sampleData[dataset])
      } else {
        if (active) setInfo(null)
      }
    }

    loadInfo()
    return () => {
      active = false
    }
  }, [dataset])


  if (!dataset) {
    return <p>Select a dataset to see details.</p>
  }

  if (!info) {
    return <p>Loading dataset info...</p>
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight text-blue-700">
          {info.name}
        </h2>
        <span className="text-sm bg-blue-100 text-blue-700 px-3 py-1 rounded-full border border-blue-200">
      {info.type}
    </span>
      </div>

      {/* Description */}
      <p className="text-zinc-600 leading-relaxed">{info.description}</p>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-3 text-sm mt-2">
        <div className="flex flex-col items-center justify-center rounded-xl bg-blue-50/60 p-3 border border-blue-100 shadow-sm">
      <span className="text-xs text-zinc-500 uppercase tracking-wide">
        Train Samples
      </span>
          <span className="text-lg font-semibold text-blue-700">
        {info.trainCount.toLocaleString()}
      </span>
        </div>
        <div className="flex flex-col items-center justify-center rounded-xl bg-purple-50/60 p-3 border border-purple-100 shadow-sm">
      <span className="text-xs text-zinc-500 uppercase tracking-wide">
        Test Samples
      </span>
          <span className="text-lg font-semibold text-purple-700">
        {info.testCount.toLocaleString()}
      </span>
        </div>
      </div>

      {/*{info.preview && (*/}
      {/*  <div className="mt-4 relative group aspect-square">*/}
      {/*    <Image*/}
      {/*      src={info.preview}*/}
      {/*      alt={`${info.name} preview`}*/}
      {/*      fill*/}
      {/*      className="rounded-2xl border border-zinc-200 object-cover shadow-md transition-transform duration-300 group-hover:scale-[1.02]"*/}
      {/*      sizes="(max-width: 768px) 100vw, 400px"*/}
      {/*    />*/}
      {/*    <span className="absolute bottom-2 right-3 text-xs text-zinc-500 bg-white/70 px-2 py-0.5 rounded-md backdrop-blur-sm">*/}
      {/*      Preview*/}
      {/*    </span>*/}
      {/*  </div>*/}
      {/*)}*/}
    </div>

  )
}

export default DatasetInfoCard
