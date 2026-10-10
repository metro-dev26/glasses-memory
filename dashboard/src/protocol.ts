// The server <-> UI contract, docs/dashboard-brief.md section 4.
// Times are Unix seconds; boxes are [x1, y1, x2, y2] in the frame's pixels.

export type Box = [number, number, number, number];

export interface Detection {
  track_id: number;
  object_id: number | null; // null while a new track is being identified
  label: string;
  name: string;
  conf: number;
  box: Box;
  reidentified: boolean;
  last_seen_ago_s?: number;
}

export interface FrameMsg {
  type: "frame";
  seq: number;
  ts: number;
  width: number;
  height: number;
  jpeg_b64: string;
  detections: Detection[];
  stats: { fps: number; latency_ms: number; in_view: number; in_memory: number };
}

export interface MemoryObject {
  object_id: number;
  name: string;
  label: string;
  last_seen: number;
  where: string;
  thumb: string;
  in_view: boolean;
}

export interface MemoryMsg {
  type: "memory";
  objects: MemoryObject[];
}

export interface Sighting {
  start: number;
  end: number;
  where: string;
}

export interface AnswerMsg {
  type: "answer";
  query: string;
  found: boolean;
  object_id: number | null;
  text: string;
  snapshot: string | null;
  box: Box | null;
  nearby: string[];
  sightings: Sighting[];
  candidates: string[];
}

export type ChangeKind = "missing" | "new" | "moved";

export interface ChangeMsg {
  type: "change";
  round_id: number;
  kind: ChangeKind;
  object_id: number;
  name: string;
  text: string;
  before: string;
  after: string;
}

export type RoundMsg =
  | { type: "round"; state: "running"; round_id: number; name: string; started: number }
  | { type: "round"; state: "ended"; round_id: number; report: ChangeMsg[] };

export type ServerMsg = FrameMsg | MemoryMsg | AnswerMsg | RoundMsg | ChangeMsg;

export type ClientMsg =
  | { type: "ask"; query: string }
  | { type: "ask_photo"; jpeg_b64: string }
  | { type: "rename"; object_id: number; name: string }
  | { type: "round_start"; name: string }
  | { type: "round_end" }
  | { type: "forget_all" };
