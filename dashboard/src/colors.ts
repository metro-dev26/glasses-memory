// Colour comes from the object id, so an object keeps its colour all session.
// Light, saturated hues: they stay readable over any video.
const PALETTE = [
  "#5ad1e6", "#f2b84b", "#7ee08a", "#f07e94", "#a99bff",
  "#4fd6b2", "#ff9a5c", "#e7e36b", "#6fb2ff", "#d78cf0",
];
const PENDING = "#b8bec7"; // a track that has no object id yet

export function idColor(objectId: number | null): string {
  return objectId === null ? PENDING : PALETTE[objectId % PALETTE.length];
}
