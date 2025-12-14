import React from 'react';
import DefenseSelect from "@/components/defense/defense-select";
import {DefenseInfo} from "@/types";
import DefenseInputs from "@/components/defense/defense-inputs";

const DefenseCard = async () => {
  const data = await fetch('http://localhost:3004/defenses')
  const defenses = await data.json() as DefenseInfo[]

  return (
    <div className="card-main flex-1 w-full max-w-2xl">
      <h2 className="text-xl font-semibold text-purple-700 mb-3">
        Defense Configuration
      </h2>
      <p className="text-zinc-600 mb-4">
        Select a defense strategy to mitigate adversarial attacks.
      </p>
      <div className="flex flex-col gap-4 text-zinc-700 ">
        <DefenseSelect defenses={defenses} />
        <DefenseInputs />
      </div>
    </div>
  );
};

export default DefenseCard;