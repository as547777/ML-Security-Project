"use client"

import MainContainer from "@/components/main-container";
import {StepNavigation} from "@/components/step-navigation";
import AttackSelect from "@/components/attack-select";
import DefenseSelect from "@/components/defense-select";

export default function AttackDefensePage() {
  return (
    <MainContainer>
      <h1 className="text-4xl md:text-5xl header-text mb-7 p-1">
        Attacks & Defenses
      </h1>

      <div className="card-main w-full max-w-2xl">
        <form className="flex flex-col gap-4 text-zinc-700">
          <AttackSelect />
          <DefenseSelect />
          <StepNavigation prev={'/dataset'} next={'/overview'} />
        </form>
      </div>
    </MainContainer>
  )
}