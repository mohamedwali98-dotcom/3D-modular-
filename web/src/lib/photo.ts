// Photos are made ready on the phone: a 48 MP picture or a HEIC/WebP file would otherwise cross mobile data in
// full, only to be refused by the server (JPEG or PNG, 10 MB, its pixel cap).

export const MAX_SIDE = 4000;              // px on the long side: far past what the outline and reading need
const MAX_BYTES = 10 * 1024 * 1024;        // the server's per-image limit
const SENDABLE = ['image/jpeg', 'image/png'];

/** The size to redraw a photo at as a JPEG, or null when it can be sent as it is. */
export function photoPlan(type: string, bytes: number, width: number, height: number): { width: number; height: number } | null {
  const k = Math.min(1, MAX_SIDE / Math.max(width, height));
  if (k === 1 && SENDABLE.includes(type) && bytes <= MAX_BYTES) return null;
  return { width: Math.round(width * k), height: Math.round(height * k) };
}

/** The photo, redrawn as a JPEG when photoPlan says so. One the browser cannot decode is kept as it is: the server
 * then says why it cannot read it. */
export async function readyPhoto(file: File): Promise<File> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    return file;
  }
  const plan = photoPlan(file.type, file.size, bitmap.width, bitmap.height);
  if (!plan) {
    bitmap.close();
    return file;
  }
  const canvas = document.createElement('canvas');
  canvas.width = plan.width;
  canvas.height = plan.height;
  canvas.getContext('2d')?.drawImage(bitmap, 0, 0, plan.width, plan.height);
  bitmap.close();
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
  if (!blob) return file;
  return new File([blob], `${file.name.replace(/\.[^.]+$/, '')}.jpg`, { type: 'image/jpeg', lastModified: file.lastModified });
}
