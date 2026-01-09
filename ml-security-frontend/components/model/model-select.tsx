'use client'

import React, {useEffect, useState} from 'react';
import ModelDetails from "@/components/model/model-details";
import FieldInput from "@/components/field-input";
import {ModelInfo} from "@/types";
import {useData} from "@/context/DataContext";
import {Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, DialogTrigger} from "@/components/ui/dialog";
import {ScrollArea} from "@/components/ui/scroll-area";
import {Button} from "@/components/ui/button";

const ModelSelect = ({ modelFamilies }: { modelFamilies: ModelInfo[] }) => {
  const { dataset, model, setModel, modelFamily, setModelFamily } = useData()
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState("")
  const [filtered, setFiltered] = useState(modelFamilies)
  const [selected, setSelected] = useState(modelFamily)

  const selectable = dataset !== null

  // Debounced search
  useEffect(() => {
    const handler = setTimeout(() => {
      const term = search.toLowerCase()
      setFiltered(
        modelFamilies.filter((mf) => mf.name.toLowerCase().includes(term))
      )
    }, 200)
    return () => clearTimeout(handler)
  }, [modelFamilies, search])

  const handleConfirm = () => {
    setModelFamily(selected)
    if (selected)
      setModel(selected.models[0])
    setOpen(false)
  }

  return (
    <div>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger className={'text-left w-full'} disabled={!selectable}>
          <ModelDetails clickable={selectable} selectable={selectable} />
        </DialogTrigger>

        <DialogContent className="max-w-lg text-zinc-700">
          <DialogHeader>
            <DialogTitle>Select a Model Family</DialogTitle>
          </DialogHeader>

          {/* Search input */}
          <input
            placeholder="Search families..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="mb-3"
          />

          {/* Scrollable dataset list */}
          <ScrollArea className="h-116 pr-2">
            <div className="space-y-3">
              {filtered.map((mf) => (
                <div
                  key={mf.name}
                  onClick={() => setSelected(mf)}
                  className={`cursor-pointer rounded-lg border p-3 transition-colors ${
                    selected === mf
                      ? "border-blue-500 bg-blue-50"
                      : "border-zinc-200 hover:bg-zinc-50"
                  }`}
                >
                  <h3 className="font-semibold text-blue-700">{mf.name}</h3>
                  <p className="text-sm text-zinc-600">{mf.description}</p>
                  <div className="mt-2 text-xs text-zinc-500">
                    <p>Category: {mf.category}</p>
                    <p>Use case: {mf.use_case}</p>
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

      <div className={'block mt-4'}>
        <FieldInput
          label={'Model'}
          tooltip={'Choose a specific model from the selected family'}
          type={'select'}
          options={modelFamily?.models || []}
          value={model}
          setValue={setModel} />
      </div>
    </div>
  );
};

export default ModelSelect;