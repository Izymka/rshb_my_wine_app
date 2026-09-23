"""PostgreSQL catalogue models and read repository.

Images stay on the filesystem; the database owns their identity, provenance and paths.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


wine_grapes = Table(
    "wine_grapes",
    Base.metadata,
    Column("wine_id", ForeignKey("wines.id", ondelete="CASCADE"), primary_key=True),
    Column("grape_id", ForeignKey("grape_varieties.id", ondelete="RESTRICT"), primary_key=True),
)


class Wine(Base):
    __tablename__ = "wines"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    producer: Mapped[str] = mapped_column(String(500), default="")
    name: Mapped[str] = mapped_column(String(1000))
    region: Mapped[str] = mapped_column(String(500), default="")
    vintage: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
    grapes: Mapped[list[GrapeVariety]] = relationship(secondary=wine_grapes, lazy="selectin")
    images: Mapped[list[WineImage]] = relationship(
        back_populates="wine", cascade="all, delete-orphan"
    )


class GrapeVariety(Base):
    __tablename__ = "grape_varieties"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)


class WineImage(Base):
    __tablename__ = "wine_images"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    wine_id: Mapped[int] = mapped_column(ForeignKey("wines.id", ondelete="CASCADE"), index=True)
    storage_path: Mapped[str] = mapped_column(Text, unique=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    mime_type: Mapped[str] = mapped_column(String(100), default="image/png")
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(50), default="catalog")
    wine: Mapped[Wine] = relationship(back_populates="images")
    derivatives: Mapped[list[ImageDerivative]] = relationship(
        back_populates="source_image", cascade="all, delete-orphan"
    )


class ImageDerivative(Base):
    __tablename__ = "image_derivatives"
    __table_args__ = (UniqueConstraint("source_image_id", "kind", "pipeline_version"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_image_id: Mapped[int] = mapped_column(ForeignKey("wine_images.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(50), default="label_square")
    pipeline_version: Mapped[str] = mapped_column(String(100))
    storage_path: Mapped[str] = mapped_column(Text, unique=True)
    sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(30), default="ready")
    bottle_box: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    label_box: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    photometric: Mapped[dict] = mapped_column(JSON, default=dict)
    source_image: Mapped[WineImage] = relationship(back_populates="derivatives")


class IndexBuild(Base):
    __tablename__ = "index_builds"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    build_id: Mapped[str] = mapped_column(String(100), unique=True)
    storage_path: Mapped[str] = mapped_column(Text)
    preprocessing_version: Mapped[str] = mapped_column(String(100))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


def database_url() -> str:
    value = os.environ.get("DATABASE_URL")
    if not value:
        raise RuntimeError("DATABASE_URL is required for PostgreSQL catalogue access")
    return value


def session_factory(url: str | None = None):
    return sessionmaker(create_engine(url or database_url(), future=True), expire_on_commit=False)


def catalog_items_from_db(url: str | None = None):
    """Return retrieval records; only prepared, valid label derivatives are indexable."""
    from .catalog import CatalogItem
    from .image_preprocess import LABEL_PREPROCESS_VERSION

    factory = session_factory(url)
    with factory() as session:
        rows = (
            session.query(Wine, WineImage, ImageDerivative)
            .join(WineImage, Wine.images)
            .join(ImageDerivative, WineImage.derivatives)
            .filter(
                ImageDerivative.kind == "label_square",
                ImageDerivative.pipeline_version == LABEL_PREPROCESS_VERSION,
                ImageDerivative.status == "ready",
            )
            .order_by(Wine.slug)
            .all()
        )
        return [
            CatalogItem(
                item_id=wine.slug,
                image_path=Path(derivative.storage_path),
                payload={
                    "name": wine.name,
                    "winery": wine.producer,
                    "region": wine.region,
                    "grapes": ", ".join(grape.name for grape in wine.grapes),
                    "vintage": wine.vintage,
                    "slug": wine.slug,
                    "source_image_path": image.storage_path,
                    "prepared_label": True,
                },
            )
            for wine, image, derivative in rows
        ]
