export const TRAFFIC_CLASSES = ["video", "voice", "file_transfer"] as const;

export const classLabels: Record<string, string> = {
  video: "Видео",
  voice: "Голосовой звонок",
  file_transfer: "Передача файла",
};

export const shortClassLabels: Record<string, string> = {
  video: "Видео",
  voice: "Звонок",
  file_transfer: "Файл",
};
