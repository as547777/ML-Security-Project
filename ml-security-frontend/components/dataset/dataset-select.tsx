"use client"

import * as React from "react"
import { useState, useEffect } from "react"
import { useData } from "@/context/DataContext"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import DatasetTypeIcon from "@/components/dataset/dataset-type-icon";
import {DatasetInfo} from "@/types";

export default function DatasetSelect({ datasets }: { datasets: DatasetInfo[] }) {
  const { dataset, setDataset } = useData()
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState("")
  const [filtered, setFiltered] = useState(datasets)
  const [selected, setSelected] = useState(dataset)

  // Debounced search
  useEffect(() => {
    const handler = setTimeout(() => {
      const term = search.toLowerCase()
      setFiltered(
        datasets.filter((d) => d.name.toLowerCase().includes(term))
      )
    }, 200)
    return () => clearTimeout(handler)
  }, [datasets, search])

  const handleConfirm = () => {
    setDataset(selected)
    setOpen(false)
  }

  return (
    <div>
      <label className="block text-sm font-medium mb-1 text-zinc-500">Dataset</label>

      {/* Trigger button */}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>
          <Button
            variant="outline"
            className="w-full justify-baseline text-zinc-700 text-md py-5"
          >
            <DatasetTypeIcon type={dataset?.type} />
            {dataset?.name || "Select a dataset"}
          </Button>
        </DialogTrigger>

        <DialogContent className="max-w-lg text-zinc-700">
          <DialogHeader>
            <DialogTitle>Select a Dataset</DialogTitle>
          </DialogHeader>

          {/* Search input */}
          <input
            placeholder="Search datasets..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="mb-3"
          />

          {/* Scrollable dataset list */}
          <ScrollArea className="h-116 pr-2">
            <div className="space-y-3">
              {filtered.map((d) => (
                <div
                  key={d.name}
                  onClick={() => setSelected(d)}
                  className={`cursor-pointer rounded-lg border p-3 transition-colors ${
                    selected === d
                      ? "border-blue-500 bg-blue-50"
                      : "border-zinc-200 hover:bg-zinc-50"
                  }`}
                >
                  <h3 className="font-semibold text-blue-700">{d.name}</h3>
                  <p className="text-sm text-zinc-600">{d.description}</p>
                  <div className="mt-2 text-xs text-zinc-500">
                    <p>Type: {d.type}</p>
                    <p>Train examples: {d.trainCount.toLocaleString()}</p>
                    <p>Test examples: {d.testCount.toLocaleString()}</p>
                  </div>
                </div>
              ))}

              {filtered.length === 0 && (
                <p className="text-sm text-zinc-500 text-center py-6">
                  No datasets found.
                </p>
              )}
            </div>
          </ScrollArea>

          {/* Footer buttons */}
          <DialogFooter className="mt-4 flex justify-between">
            <Button variant="ghost" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleConfirm} disabled={!selected}>
              Confirm
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
