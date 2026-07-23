from pydantic import BaseModel, Field, model_validator


class ModelInfo(BaseModel):
    id: str
    name: str
    active: bool
    loaded: bool
    task_type: str


class ModelClassInfo(BaseModel):
    """
    Clase perteneciente al catálogo de un modelo de clasificación.
    """

    index: int = Field(ge=0)
    label: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)


class ModelClassCatalog(BaseModel):
    """
    Catálogo completo y ordenado de clases del modelo activo.
    """

    model_id: str
    model_name: str
    total_classes: int = Field(ge=1)
    classes: list[ModelClassInfo] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_class_count(self):
        if self.total_classes != len(self.classes):
            raise ValueError("Total classes must match the class catalog")

        indices = [class_info.index for class_info in self.classes]

        if len(indices) != len(set(indices)):
            raise ValueError("Model class indices must be unique")

        return self
