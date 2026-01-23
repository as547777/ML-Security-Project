"use client"

import MainContainer from "@/components/main-container";
import React, {useState} from "react";
import {ResultInfo} from "@/types";
import ParamCard from "@/components/param-card";
import Card from "@/components/card";
import PoisonExample from "@/components/results/poison-example";
import Section from "@/components/section";

export default function ResultsPage() {
  const [resultsData, setResultsData] = useState<ResultInfo>(() => {
    if (typeof window === "undefined") return null;

    try {
      const data = sessionStorage.getItem("experimentResults");
      return data ? JSON.parse(data) : null;
    } catch {
      return null;
    }
  });

  return (
    <MainContainer>
      <h1 className="text-4xl md:text-5xl header-text mb-7 p-1">
        Experiment Results
      </h1>

      {resultsData ? (
        <div className={'w-full'}>
          <Card
            title={"Core Effectiveness Metrics"}
            description={"Key metrics measuring the effectiveness of the backdoor attack and defense mechanisms across experimental phases."}
            fullWidth
            className={'flex flex-row gap-6 items-start justify-between'}
          >
            <Section title={"Attack Phase"}>
              <div className="flex flex-col gap-4 text-zinc-700">
                <div className={"grid grid-cols-2 gap-2"}>
                  <ParamCard label={"Clean Performance (CP)"} value={resultsData.attack_phase.accuracy} />
                  <ParamCard label={"Attack Success Rate (ASR)"} value={resultsData.attack_phase.asr} />
                </div>
                <PoisonExample examples={resultsData.visualizations} />
              </div>
            </Section>

            <Section title={"Defense Phase"}>
              <div className={"grid grid-cols-2 gap-2"}>
                <ParamCard label={"Accuracy"} value={resultsData.defense_phase.accuracy} />
                <ParamCard label={"Attack Success Rate"} value={resultsData.defense_phase.asr} />
              </div>
            </Section>

            <Section title={"Experiment Summary"}>
              <div className="grid grid-cols-2 gap-2">
                <ParamCard label={"Clean Accuracy Drop (CAD)"} value={resultsData.improvement.asr_reduction} />
                <ParamCard label={"Defense Residual ASR (rASR)"} value={resultsData.improvement.asr_reduction} highlight />
              </div>
            </Section>
          </Card>
        </div>
        ) : (
        <p>No results yet.</p>
      )}
    </MainContainer>
  )
}