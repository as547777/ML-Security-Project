import React from 'react';
import AttackSelect from "@/components/attack-select";
import AttackInputs from "@/components/attack-inputs";
import {AttackInfo} from "@/types";

const AttackCard = async () => {
  const data = await fetch('http://localhost:3004/attacks')
  const attacks = await data.json() as AttackInfo[]

  return (
    <div className="card-main flex-1 w-full max-w-2xl">
      <h2 className="text-xl font-semibold text-blue-700 mb-3">
        Attack Configuration
      </h2>
      <p className="text-zinc-600 mb-4">
        Choose an adversarial attack method to test your model’s robustness.
      </p>
      <div className="flex flex-col gap-4 text-zinc-700 ">
        <AttackSelect attacks={attacks} />
        {/* temp */}
        <AttackInputs />
      </div>
    </div>
  );
};

export default AttackCard;