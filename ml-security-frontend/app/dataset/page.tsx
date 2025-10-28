"use client"

import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"

export default function DatasetPage() {
  return (
    <motion.main
      className="flex flex-col items-center justify-center min-h-screen bg-zinc-50 p-8"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <h1 className="text-3xl font-bold text-zinc-800 mb-6">
        Odabir dataseta i hiperparametara
      </h1>

      <div className="bg-white shadow-md rounded-2xl p-6 w-full max-w-xl">
        <p className="text-zinc-600 mb-4">
          Ovdje možeš odabrati dataset koji ćeš koristiti i postaviti osnovne hiperparametre mreže.
        </p>

        <form className="flex flex-col gap-4">
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1">
              Dataset
            </label>
            <select className="w-full border border-zinc-300 rounded-md p-2 focus:outline-none focus:ring-2 focus:ring-zinc-400">
              <option>MNIST</option>
              <option>CIFAR-10</option>
              <option>Custom Dataset</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1">
              Learning Rate
            </label>
            <input
              type="number"
              step="0.001"
              defaultValue="0.01"
              className="w-full border border-zinc-300 rounded-md p-2 focus:outline-none focus:ring-2 focus:ring-zinc-400"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1">
              Broj epoha
            </label>
            <input
              type="number"
              defaultValue="10"
              className="w-full border border-zinc-300 rounded-md p-2 focus:outline-none focus:ring-2 focus:ring-zinc-400"
            />
          </div>

          <Button type="submit" className="mt-4">
            Nastavi
          </Button>
        </form>
      </div>
    </motion.main>
  )
}