import { expect, type Request, test } from "@playwright/test";

import type { Prediction } from "../src/api/client";
import { classLabels, TRAFFIC_CLASSES } from "../src/trafficClasses";

const url = process.env.TRAFFICCLASSIFIER_WEB_URL || "http://127.0.0.1:8000/app/";
const percent = new Intl.NumberFormat("ru-RU", { style: "percent", maximumFractionDigits: 1 });
const demo = {
  offsets_us: [0, 2000, 5250],
  sizes_bytes: [100, 300, 180],
  directions: [1, 0, 1],
};

test.use({ channel: "chrome", viewport: { width: 360, height: 900 } });

function isPredictionRequest(request: Request): boolean {
  return request.method() === "POST" && new URL(request.url()).pathname === "/api/v1/predict";
}

test("На мобильном экране пример проходит через UI и возвращает прогноз CNN", async ({ page }) => {
  let requests = 0;
  page.on("request", (request) => {
    if (isPredictionRequest(request)) {
      requests += 1;
    }
  });

  await page.goto(url);
  await expect(page.getByText("Модель готова", { exact: true })).toBeVisible();

  const predict = page.getByRole("button", { name: "Распознать", exact: true });
  await predict.click();
  await expect(page.getByText("Исправьте отмеченные поля.")).toBeVisible();
  await expect(page.getByRole("group", { name: "Пакет 1" }).getByLabel("Размер, байт")).toBeFocused();

  await page.getByRole("button", { name: "Загрузить пример" }).press("Enter");
  const response = page.waitForResponse((reply) => isPredictionRequest(reply.request()));
  await predict.press("Enter");

  const reply = await response;
  expect(reply.status()).toBe(200);
  expect(reply.request().postDataJSON()).toEqual(demo);
  const prediction: Prediction = await reply.json();
  const panel = page.getByRole("region", { name: "Ответ модели" });
  await expect(panel.getByRole("status")).toHaveText(`Предсказанный тип: ${classLabels[prediction.category]}`);
  await expect(panel.getByText(`${prediction.observed_packets} из ${prediction.n} пакетов`)).toBeVisible();
  await expect(panel.getByText(prediction.model_id, { exact: true })).toBeVisible();
  const scores = panel.getByRole("listitem");
  await expect(scores).toHaveCount(TRAFFIC_CLASSES.length);
  for (const [index, category] of TRAFFIC_CLASSES.entries()) {
    await expect(scores.nth(index)).toContainText(classLabels[category]);
    await expect(scores.nth(index)).toContainText(percent.format(prediction.scores[category]));
  }
  expect(requests).toBe(1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
