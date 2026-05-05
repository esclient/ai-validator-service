import pytest
from faker import Faker

from aivalidatorservice.service.service import ModerationService

fake = Faker()
fake_fr = Faker("fr_FR")
fake_ru = Faker("ru_RU")
fake_ja = Faker("ja_JP")


@pytest.mark.asyncio
async def test_high_logit_returns_true(
    toxic_service: ModerationService,
) -> None:
    result = await toxic_service.moderate(fake.text())
    assert result is True


@pytest.mark.asyncio
async def test_low_logit_returns_false(
    clean_service: ModerationService,
) -> None:
    result = await clean_service.moderate(fake.text())
    assert result is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "",
        fake.sentence(),
        fake.text(),
        fake_fr.sentence(),
        fake_ru.sentence(),
        fake_ja.sentence(),
    ],
)
async def test_crash_safety_inputs_return_boolean(
    clean_service: ModerationService, text: str
) -> None:
    result = await clean_service.moderate(text)
    assert isinstance(result, bool)
