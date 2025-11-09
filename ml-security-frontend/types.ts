export interface DatasetInfo {
  name: string
  description: string
  type: string
  trainCount: number
  testCount: number
}

export interface AttackInfo {
  name: string
  description: string
  type: string
  // TODO - pitat piceka sta bi jos mogli dodat da opisemo napad
  //
  // parameters: {
  //   [key: string]: string | number | boolean
  // }
}

// TODO - napravit isto to samo za DefenseInfo
