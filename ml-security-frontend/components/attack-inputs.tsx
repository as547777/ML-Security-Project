import React from 'react';
import FieldInput from "@/components/field-input";
import {useData} from "@/context/DataContext";

const AttackInputs = () => {
  const {
    sourceLabel, setSourceLabel,
    targetLabel, setTargetLabel,
    poisonRate, setPoisonRate,
    triggerSize, setTriggerSize,
  } = useData()

  return (
    <div className="flex flex-wrap gap-4">
      <FieldInput
        label="Source label"
        tooltip="Label of the class that will be poisoned (e.g., 1)"
        type="number"
        step={1}
        value={sourceLabel}
        setValue={setSourceLabel}
      />

      <FieldInput
        label="Target label"
        tooltip="Label of the class that poisoned samples should be misclassified as (e.g., 7)"
        type="number"
        step={1}
        value={targetLabel}
        setValue={setTargetLabel}
      />

      <FieldInput
        label="Poison rate"
        tooltip="Fraction of samples from the source class to poison (0–1)"
        type="number"
        step={0.1}
        value={poisonRate}
        setValue={setPoisonRate}
      />

      <FieldInput
        label="Trigger size"
        tooltip="Size of the injected trigger patch (e.g., 4 for a 4×4 pixel square)"
        type="number"
        step={1}
        value={triggerSize}
        setValue={setTriggerSize}
      />
    </div>
  );
};

export default AttackInputs;