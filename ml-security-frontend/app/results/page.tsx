"use client"

import MainContainer from "@/components/main-container";
import React, {useState} from "react";
import {ResultInfo} from "@/types";
import ParamCard from "@/components/param-card";
import Image from "next/image";
import Section from "@/components/section";
import Card from "@/components/card";
import CardContainer from "@/components/card-container";

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
        Test Results
      </h1>

      {resultsData ? (
        <CardContainer>
          <Card
            title={"Attack Phase"}
            description={"Information on the first phase of the experiment, where the model is trained with poisoned data."}
          >
            <div className="flex flex-col gap-4 text-zinc-700">
              <div className={"grid grid-cols-2 gap-2"}>
                <ParamCard label={"Model accuracy"} value={resultsData.attack_phase.accuracy} />
                <ParamCard label={"Attack Success Ratio"} value={resultsData.attack_phase.asr} />
              </div>
              <Section title={"Class poison visualization"}>
                <div className={'flex items-center justify-between mb-2 px-15'}>
                  <Image
                    src={`data:image/png;base64,${resultsData.visualizations[0].original_image}`}
                    alt={"Original Image"}
                    width={100}
                    height={100}
                  />
                  <span className={'text-2xl'}>→</span>
                  <Image
                    src={`data:image/png;base64,${resultsData.visualizations[0].poisoned_image}`}
                    alt={"Poisoned Image"}
                    width={100}
                    height={100}
                  />
                </div>
              </Section>
            </div>
          </Card>

          <Card
            title={"Defense Phase"}
            description={"Information on the second phase of the experiment, where the model is trained without poisoned data."}>
            <div className={"grid grid-cols-2 gap-2"}>
              {/*<ParamCard label={"Pruned Accuracy"} value={resultsData.defense_phase.acc_pruned} />*/}
              {/*<ParamCard label={"Pruned Attack Success Ratio"} value={resultsData.defense_phase.asr_pruned} />*/}
              <ParamCard label={"Accuracy"} value={resultsData.defense_phase.accuracy} />
              <ParamCard label={"Attack Success Ratio"} value={resultsData.defense_phase.asr} />
            </div>
          </Card>

          <Card title={"Experiment Summary"} description={"Summary of the experiment results."}>
            <div className="grid grid-cols-2 gap-2">
              <ParamCard label={"Accuracy Reduction"} value={resultsData.improvement.asr_reduction} />
              <ParamCard label={"Attack Success Rate Reduction"} value={resultsData.improvement.asr_reduction} highlight />
            </div>
          </Card>
        </CardContainer>
        ) : (
        <p>No results yet.</p>
      )}
    </MainContainer>
  )
}