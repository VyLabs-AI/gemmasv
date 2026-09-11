const PNG_SIGNATURE = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
]);
const COLOR_CHUNKS = new Set(["sRGB", "iCCP", "gAMA", "cHRM"]);

const CRC_TABLE = Array.from({ length: 256 }, (_, value) => {
  let crc = value;
  for (let bit = 0; bit < 8; bit += 1) {
    crc = (crc & 1) !== 0 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
  }
  return crc >>> 0;
});

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function encodeChunk(type, payload) {
  const name = Buffer.from(type, "ascii");
  const length = Buffer.alloc(4);
  length.writeUInt32BE(payload.length);
  const checksum = Buffer.alloc(4);
  checksum.writeUInt32BE(crc32(Buffer.concat([name, payload])));
  return Buffer.concat([length, name, payload, checksum]);
}

export function parsePngChunks(buffer) {
  if (!buffer.subarray(0, 8).equals(PNG_SIGNATURE)) {
    throw new Error("not a PNG");
  }
  const chunks = [];
  let offset = 8;
  while (offset < buffer.length) {
    if (offset + 12 > buffer.length) throw new Error("truncated PNG chunk");
    const length = buffer.readUInt32BE(offset);
    const type = buffer.subarray(offset + 4, offset + 8).toString("ascii");
    const payloadEnd = offset + 8 + length;
    const payload = buffer.subarray(offset + 8, payloadEnd);
    const expected = buffer.readUInt32BE(payloadEnd);
    const actual = crc32(Buffer.concat([Buffer.from(type, "ascii"), payload]));
    if (expected !== actual) throw new Error(`invalid PNG CRC for ${type}`);
    chunks.push({ type, payload });
    offset = payloadEnd + 4;
    if (type === "IEND") break;
  }
  if (chunks.at(0)?.type !== "IHDR" || chunks.at(-1)?.type !== "IEND") {
    throw new Error("invalid PNG chunk order");
  }
  return chunks;
}

export function normalizePngSrgb(buffer) {
  const chunks = parsePngChunks(buffer).filter(
    ({ type }) => !COLOR_CHUNKS.has(type),
  );
  chunks.splice(1, 0, { type: "sRGB", payload: Buffer.from([0]) });
  return Buffer.concat([
    PNG_SIGNATURE,
    ...chunks.map(({ type, payload }) => encodeChunk(type, payload)),
  ]);
}
