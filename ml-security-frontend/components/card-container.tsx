import React, {PropsWithChildren} from 'react';

const CardContainer = ({children}: PropsWithChildren) => {
  return (
    <div className={'flex flex-col lg:flex-row gap-6 items-start'}>
      {children}
    </div>
  );
};

export default CardContainer;