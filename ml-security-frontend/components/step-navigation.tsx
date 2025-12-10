"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowLeft, Play } from "lucide-react"
import React from "react"

interface StepNavigationProps {
  isFinal?: boolean
  onRun?: () => void
  prev?: string
  next?: string
  isRunning?: boolean
}

export function StepNavigation({
                                 isFinal = false,
                                 onRun,
                                 prev = "/",
                                 next = "/",
                                 isRunning
                               }: StepNavigationProps) {
  return (
    <div className="flex justify-between mt-4 w-full">
      <Link href={prev}>
        <Button
          variant="outline"
          className="flex items-center gap-2 border border-zinc-300 text-zinc-700 hover:bg-zinc-100 hover:text-zinc-900"
        >
          <ArrowLeft className="w-4 h-4" />
          Go back
        </Button>
      </Link>

      {isFinal ? (
        <Button
          onClick={onRun}
          disabled={isRunning}
          className={`flex items-center gap-2 text-white ${
            isRunning
              ? "bg-green-400 cursor-not-allowed"
              : "bg-green-600 hover:bg-green-700"
          }`}
        >
          <Play className="w-4 h-4" />
          {isRunning ? "Running..." : "Run Test"}
        </Button>

      ) : (
        <Link href={next}>
          <Button className="bg-black text-white hover:bg-zinc-800">
            Next step
          </Button>
        </Link>
      )}
    </div>
  )
}
