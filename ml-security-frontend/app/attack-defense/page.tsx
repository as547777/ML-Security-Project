import MainContainer from "@/components/main-container"
import { StepNavigation } from "@/components/step-navigation"
import AttackCard from "@/components/attack/attack-card";
import DefenseCard from "@/components/defense/defense-card";
import CardContainer from "@/components/card-container";

export default async function AttackDefensePage() {
  return (
    <MainContainer>
      <h1 className="text-4xl md:text-5xl header-text mb-7 p-1">
        Attacks & Defenses
      </h1>

      {/* Container for the two cards */}
      <CardContainer>
        <AttackCard />
        <DefenseCard />
      </CardContainer>

      {/* Step navigation under both cards */}
      <div className="mt-8 flex justify-center w-full max-w-sm">
        <StepNavigation prev="/dataset" next="/overview" />
      </div>
    </MainContainer>
  )
}
