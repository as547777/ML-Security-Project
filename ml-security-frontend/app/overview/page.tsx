"use client"

import MainContainer from "@/components/main-container"
import { StepNavigation } from "@/components/step-navigation"
import { useData } from "@/context/DataContext"
import React, {useState} from "react";
import {ParamsType} from "@/types";
import Section from "@/components/section";
import ParamCard from "@/components/param-card";
import { useRouter } from 'next/navigation'
import DatasetDetails from "@/components/dataset/dataset-details";
import Card from "@/components/card";
import ModelDetails from "@/components/model/model-details";
import AttackDetails from "@/components/attack/attack-details";
import DefenseDetails from "@/components/defense/defense-details";

export default function OverviewPage() {
  const API_URL = process.env.NEXT_PUBLIC_API_URL_CLIENT || 'http://localhost:5000';

  const { dataset, momentum, batchSize, runCount, seed, learningRate, optimizer, lossFunction,
    epochs, attack, attackParams, defense, defenseParams, model } = useData()
  const [isRunning, setIsRunning] = useState(false)
  const selectable = dataset !== null

  const router = useRouter();

  const extractParamValues = (params?: ParamsType) => {
    if (!params) return {};

    return Object.entries(params).reduce((acc, [key, param]) => {
      acc[key] = param.value;
      return acc;
    }, {} as Record<string, number | string>);
  };

  const renderParams = (params?: ParamsType) => {
    if (!params || Object.keys(params).length === 0) return null

    return (
      <>
        {Object.entries(params).map(([key, param]) => (
          <ParamCard
            key={key}
            label={param.label}
            value={param.value}
          />
        ))}
      </>
    )
  }

  // TODO - provjerit zasto proxy ne ceka response neg faila
  // TODO - prebacit u konkretnu next.js api datoteku
  const execute = async () => {
    setIsRunning(true);
    try {
      const attackParamsValues = extractParamValues(attackParams);
      const defenseParamsValues = extractParamValues(defenseParams);

      const response = await fetch(API_URL + '/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          "dataset": dataset?.name,
          "learning_rate": learningRate,
          "epochs": epochs,
          "momentum": momentum,
          "num_of_runs": runCount,
          "seed": seed,
          "attack": attack?.name,
          "model": model,
          "defense": defense?.name,
          "attack_params": attackParamsValues,
          "defense_params": defenseParamsValues,
          "metrics": ["AccuracyDifference", "CalculateStd"]
        })
      })
      if (!response.ok) {
        throw new Error('Network response was not ok')
      }
      const data = await response.json()
      sessionStorage.setItem('experimentResults', JSON.stringify(data));
      router.push('/results');
    }
    catch (error) {
      alert(error)
    }
    finally {
      setIsRunning(false);
    }
  }

  return (
    <MainContainer>
      <h1 className="text-4xl md:text-5xl header-text mb-7 p-1">
        Overview
      </h1>

      <Card fullWidth>
        <div className="space-y-6">
          <p className="text-zinc-600">
            Here’s a summary of your selected configuration:
          </p>

          <div className={'space-y-6'}>
            <Section title="Training Configuration">
              <div className="grid lg:grid-cols-2 gap-4 items-start">
                <div className="space-y-4">
                  <DatasetDetails />
                  <ModelDetails selectable={selectable} />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <ParamCard label={"Model"} value={model} highlight className={'col-span-2'} />
                  <ParamCard label="Learning Rate" value={learningRate} />
                  <ParamCard label="Epochs" value={epochs} />
                  <ParamCard label="Batch Size" value={batchSize} />
                  <ParamCard label="Momentum" value={momentum} />
                  <ParamCard label="Number of runs" value={runCount} />
                  <ParamCard label="Seed" value={seed} />
                  <ParamCard label="Optimizer" value={optimizer} />
                  <ParamCard label="Loss Function" value={lossFunction} />
                </div>
              </div>
            </Section>


            <Section title="Attack Configuration">
              <div className="grid lg:grid-cols-2 gap-4 items-start">
                <div className="space-y-4">
                  <AttackDetails selectable={selectable} />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  {renderParams(attackParams)}
                </div>
              </div>
            </Section>


            <Section title="Defense Configuration">
              <div className="grid lg:grid-cols-2 gap-4 items-start">
                <div className="space-y-4">
                  <DefenseDetails selectable={selectable} />
                </div>

                <div className="grid grid-cols-2 gap-2 items-start">
                  {renderParams(defenseParams)}
                </div>
              </div>
            </Section>

          </div>
        </div>

        <StepNavigation
          prev="/attack-defense"
          isFinal={true}
          isRunning={isRunning}
          onRun={execute}
        />
      </Card>
    </MainContainer>
  )
}
