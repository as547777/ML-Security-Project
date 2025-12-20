import React, {PropsWithChildren} from 'react';

interface Props {
  className?: string;
  title?: string;
  description?: string;
}

const Card = ({className, title, description, children}: PropsWithChildren<Props>) => {
  return (
    <div className="card-main flex-1 w-full min-w-lg max-w-2xl">
      {title && <h2 className="text-xl font-semibold text-blue-700 mb-3">{title}</h2>}
      {description && <p className="text-zinc-600 mb-4">{description}</p>}

      {children}
    </div>
  );
};

export default Card;