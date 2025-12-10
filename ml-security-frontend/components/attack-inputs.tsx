import React, {useEffect} from 'react';
import FieldInput from "@/components/field-input";
import {useData} from "@/context/DataContext";

const AttackInputs = () => {
  const {attackParams, updateAttackParams} = useData()

  useEffect(() => {
    console.log(attackParams)
  }, [attackParams])

  if (!attackParams) {
    return null;
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-2 gap-4">
      {Object.entries(attackParams).map(([key, param]) => (
        <FieldInput
          key={key}
          label={param.label}
          tooltip={param.tooltip}
          type={param.type as "number" | "string"}
          step={param.step}
          options={param.options}
          value={param.value}
          setValue={(value) => {
            updateAttackParams(key, value)
          }}
        />
      ))}
    </div>
  );
};

export default AttackInputs;