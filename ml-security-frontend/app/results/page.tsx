"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import MainContainer from "@/components/main-container";

export default function ResultsPage() {
  return (
    <MainContainer>
      <h1 className="text-4xl md:text-5xl header-text mb-7 p-1">
        Test Results
      </h1>

      <div className="card-main max-w-2xl">
        <p className="text-zinc-600 mb-4">
          Ovdje će se prikazivati konačni rezultati eksperimenta, uključujući metrike točnosti i robusnosti.
        </p>

        <div className="border border-dashed border-zinc-300 rounded-lg h-40 flex items-center justify-center text-zinc-400">
          (Rezultati / tablica / metrike)
        </div>

        <Link href="/">
          <Button className="mt-8 w-full" variant="secondary">
            Povratak na početnu
          </Button>
        </Link>
      </div>
    </MainContainer>
  )
}