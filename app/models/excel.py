from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid
from app.db.types import GUID


class ExcelWorkbook(Base, TimestampMixin):
    """One uploaded XLSX/CSV file. A CSV always has exactly one sheet."""

    __tablename__ = "excel_workbooks"

    workbook_id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(
        GUID, ForeignKey("sessions.session_id", ondelete="CASCADE"), index=True, nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    file_id: Mapped[str] = mapped_column(
        GUID, ForeignKey("files.file_id", ondelete="CASCADE"), index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)

    sheets: Mapped[list["ExcelSheet"]] = relationship(
        back_populates="workbook", cascade="all, delete-orphan"
    )


class ExcelSheet(Base, TimestampMixin):
    """One sheet/table. Backed by a dynamically-created Postgres table."""

    __tablename__ = "excel_sheets"

    sheet_id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    workbook_id: Mapped[str] = mapped_column(
        GUID, ForeignKey("excel_workbooks.workbook_id", ondelete="CASCADE"), index=True, nullable=False
    )
    session_id: Mapped[str] = mapped_column(GUID, index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)

    sheet_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Physical name of the dynamically-created table holding this sheet's rows.
    table_name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    row_count: Mapped[int] = mapped_column(Integer, default=0)

    # Free-text semantic summary of this sheet (embedded into Qdrant so
    # semantic/hybrid questions can discover the right sheet).
    semantic_summary: Mapped[str] = mapped_column(Text, default="")

    workbook: Mapped["ExcelWorkbook"] = relationship(back_populates="sheets")
    columns: Mapped[list["ExcelColumnSchema"]] = relationship(
        back_populates="sheet", cascade="all, delete-orphan"
    )


class ExcelColumnSchema(Base, TimestampMixin):
    """Inferred schema for one column of one sheet."""

    __tablename__ = "excel_schema"

    column_id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    sheet_id: Mapped[str] = mapped_column(
        GUID, ForeignKey("excel_sheets.sheet_id", ondelete="CASCADE"), index=True, nullable=False
    )

    column_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Physical (sanitized) column name in the dynamic table.
    physical_name: Mapped[str] = mapped_column(String(128), nullable=False)
    # TEXT | INTEGER | NUMERIC | BOOLEAN | DATE | TIMESTAMP
    data_type: Mapped[str] = mapped_column(String(16), nullable=False)

    is_numeric: Mapped[bool] = mapped_column(Boolean, default=False)
    is_date: Mapped[bool] = mapped_column(Boolean, default=False)
    is_text: Mapped[bool] = mapped_column(Boolean, default=False)
    is_categorical: Mapped[bool] = mapped_column(Boolean, default=False)

    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    sheet: Mapped["ExcelSheet"] = relationship(back_populates="columns")
