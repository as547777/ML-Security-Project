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

export default function OverviewPage() {
  const { dataset, momentum, batchSize, optimizer, lossFunction, learningRate, epochs, attack, attackParams, defense, defenseParams } = useData()
  const [isRunning, setIsRunning] = useState(false)

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
      <div className="grid grid-cols-2 gap-2 mt-3">
        {Object.entries(params).map(([key, param]) => (
          <ParamCard
            key={key}
            label={param.label}
            value={param.type === 'number' ? Number(param.value).toFixed(4) : param.value}
          />
        ))}
      </div>
    )
  }

  // TODO - provjerit zasto proxy ne ceka response neg faila
  // TODO - prebacit u konkretnu next.js api datoteku
  const execute = async () => {
    setIsRunning(true);
    try {
      const attackParamsValues = extractParamValues(attackParams);
      const defenseParamsValues = extractParamValues(defenseParams);

      const response = await fetch('http://localhost:5000/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          "dataset": dataset?.name,
          "learning_rate": learningRate,
          "epochs": epochs,
          "momentum": momentum,
          "attack": attack?.name,
          "model": "ImageModel",
          "defense": defense?.name,
          "attack_params": attackParamsValues,
          "defense_params": defenseParamsValues
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

      <Card>
        <div className="space-y-6">
          <p className="text-zinc-600">
            Here’s a summary of your selected configuration:
          </p>

          <DatasetDetails />

          <Section title={'Training Configuration'}>
            <div className="grid grid-cols-2 gap-2">
              <ParamCard label="Learning Rate" value={learningRate} highlight />
              <ParamCard label="Epochs" value={epochs} highlight />
              <ParamCard label="Batch Size" value={batchSize} highlight />
              <ParamCard label="Momentum" value={momentum} />
              <ParamCard label="Optimizer" value={optimizer} />
              <ParamCard label="Loss Function" value={lossFunction} />
            </div>
          </Section>

          <Section title="Attack Configuration">
            <ParamCard label="Attack Method" value={attack?.name} highlight />
            {attack?.description && (
              <p className="text-xs text-zinc-600 mt-2 mb-1">{attack.description}</p>
            )}
            {renderParams(attackParams)}
          </Section>

          <Section title="Defense Configuration">
            <ParamCard label="Defense Method" value={defense?.name} highlight />
            {defense?.description && (
              <p className="text-xs text-zinc-600 mt-2 mb-1">{defense.description}</p>
            )}
            {renderParams(defenseParams)}
          </Section>
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
