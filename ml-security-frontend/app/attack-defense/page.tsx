"use client"

import MainContainer from "@/components/main-container"
import { StepNavigation } from "@/components/step-navigation"
import AttackCard from "@/components/attack-card";
import DefenseCard from "@/components/defense-card";

export default function AttackDefensePage() {
  return (
    <MainContainer>
      <h1 className="text-4xl md:text-5xl header-text mb-7 p-1">
        Attacks & Defenses
      </h1>

      {/* Container for the two cards */}
      <div className="flex flex-col lg:flex-row gap-6">
        <AttackCard />
        <DefenseCard />
      </div>

      {/* Step navigation under both cards */}
      <div className="mt-8 flex justify-center w-full max-w-sm">
        <StepNavigation prev="/dataset" next="/overview" />
      </div>
    </MainContainer>
  )
}
