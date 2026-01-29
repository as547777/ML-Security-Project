import React from 'react';
import DefenseSelect from "@/components/defense/defense-select";
import {DefenseInfo} from "@/types";
import DefenseInputs from "@/components/defense/defense-inputs";
import Card from "@/components/card";

const DefenseCard = async () => {
  const API_URL = process.env.NEXT_PUBLIC_API_URL_SERVER || 'http://localhost:5000';

  const data = await fetch(API_URL + '/defenses')
  const defenses = await data.json() as DefenseInfo[]

  return (
    <Card>
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
    </Card>
  );
};

export default DefenseCard;