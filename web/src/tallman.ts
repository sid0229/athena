/**
 * Tall Man lettering for look-alike / sound-alike drug names (display only).
 * Source: FDA Name Differentiation Project and the ISMP list of drug names with
 * recommended tall man letters (generic names). Matching and logic never use this.
 */
const LIST = [
  'acetaZOLAMIDE', 'acetoHEXAMIDE', 'ALPRAZolam', 'LORazepam', 'clonazePAM', 'buPROPion', 'busPIRone',
  'carBAMazepine', 'OXcarbazepine', 'chlordiazePOXIDE', 'chlorproMAZINE', 'chlorproPAMIDE',
  'clomiPHENE', 'clomiPRAMINE', 'cloNIDine', 'cycloSERINE', 'cycloSPORINE',
  'DAUNOrubicin', 'DOXOrubicin', 'IDArubicin', 'epiRUBicin', 'dimenhyDRINATE', 'diphenhydrAMINE',
  'DOBUTamine', 'DOPamine', 'ePHEDrine', 'EPINEPHrine', 'FLUoxetine', 'PARoxetine', 'DULoxetine',
  'glipiZIDE', 'glyBURIDE', 'guaiFENesin', 'guanFACINE', 'hydrALAZINE', 'hydrOXYzine', 'HYDROmorphone',
  'HYDROcodone', 'oxyCODONE', 'ISOtretinoin', 'lamiVUDine', 'lamoTRIgine', 'levETIRAcetam',
  'levOCARNitine', 'levoFLOXacin', 'medroxyPROGESTERone', 'methylPREDNISolone', 'methylTESTOSTERone',
  'metFORMIN', 'metroNIDAZOLE', 'mitoXANTRONE', 'niCARdipine', 'NIFEdipine', 'niMODipine',
  'OLANZapine', 'QUEtiapine', 'PENTobarbital', 'PHENobarbital', 'prednisoLONE', 'predniSONE',
  'risperiDONE', 'rOPINIRole', 'sAXagliptin', 'SITagliptin', 'sulfADIAZINE', 'sulfaSALAzine',
  'tiaGABine', 'tiZANidine', 'TOLAZamide', 'TOLBUTamide', 'traMADol', 'traZODone', 'vinBLAStine', 'vinCRIStine',
]
const MAP = new Map(LIST.map(t => [t.toLowerCase(), t]))
const WORD = /[a-z]+/gi

/** 'hydroxyzine' -> 'hydrOXYzine'; names not on the list are returned unchanged. */
export function tallMan(s: string): string {
  return s.replace(WORD, w => MAP.get(w.toLowerCase()) ?? w)
}
