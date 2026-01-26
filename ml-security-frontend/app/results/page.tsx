"use client"

import MainContainer from "@/components/main-container";
import React, {useState} from "react";
import {ResultInfo} from "@/types";
import ParamCard from "@/components/param-card";
import Card from "@/components/card";
import PoisonExample from "@/components/results/poison-example";
import Section from "@/components/section";
import {RunGraph} from "@/components/results/run-graph";

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
        <div className="w-full">
          <Card
            title="Core Effectiveness Metrics"
            description="Detailed attack, defense, and per-run effectiveness metrics."
            fullWidth
            className="flex flex-col gap-8"
          >
            <Section title="Poison visualization">
              <PoisonExample limit={3} examples={resultsData.visualizations} />
            </Section>

            {resultsData.metrics.attack &&
              <Section title="Attack Configuration">
                <div className="grid grid-cols-2 2xl:grid-cols-4 gap-2">
                    <>
                      <ParamCard
                        label="Patch Area Ratio"
                        value={resultsData.metrics.attack.patch_area_ratio}
                      />
                      <ParamCard
                        label="Poison Rate"
                        value={resultsData.metrics.attack.poison_rate}
                      />
                    </>
                </div>
              </Section>
            }

            <Section title="Attack Phase (Before Defense)">
              <div className="grid grid-cols-2 2xl:grid-cols-4 gap-2">
                <ParamCard
                  label="Clean Accuracy (mean)"
                  value={resultsData.metrics.deviations.clean_acc.before.mean}
                />
                <ParamCard
                  label="Clean Accuracy (std)"
                  value={resultsData.metrics.deviations.clean_acc.before.std}
                />
                <ParamCard
                  label="ASR (mean)"
                  value={resultsData.metrics.deviations.asr.before.mean}
                />
                <ParamCard
                  label="ASR (std)"
                  value={resultsData.metrics.deviations.asr.before.std}
                />
              </div>
            </Section>

            <Section title="Defense Phase (After Defense)">
              <div className="grid grid-cols-2 2xl:grid-cols-4 gap-2">
                <ParamCard
                  label="Clean Accuracy (mean)"
                  value={resultsData.metrics.deviations.clean_acc.after.mean}
                />
                <ParamCard
                  label="Clean Accuracy (std)"
                  value={resultsData.metrics.deviations.clean_acc.after.std}
                />
                <ParamCard
                  label="ASR (mean)"
                  value={resultsData.metrics.deviations.asr.after.mean}
                />
                <ParamCard
                  label="ASR (std)"
                  value={resultsData.metrics.deviations.asr.after.std}
                />
              </div>
            </Section>

            <Section title="Per-Run Improvements">
              <div className="grid grid-cols-1 2xl:grid-cols-2 gap-6">
                <ParamCard
                  label="Clean Accuracy Drop per Run"
                  value={
                    <RunGraph
                      color={"bg-violet-500"}
                      values={resultsData.metrics.improvement.acc_drop}
                    />
                  }
                />
                <ParamCard
                  label="ASR Reduction per Run"
                  value={
                    <RunGraph
                      values={resultsData.metrics.improvement.asr_reduction}
                      color={"bg-emerald-500"}
                    />
                  }
                />
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