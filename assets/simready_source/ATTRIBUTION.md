# SimReady asset attribution

## Imagine-IO PhysicalAI SimReady Kitchen Assets

Source dataset: [PhysicalAI SimReady Kitchen Assets](https://huggingface.co/datasets/imagine-io/PhysicalAI-Simready-Kitchen-Assets), distributed under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).

- `mug_coffee/mug_coffee.usd` and `mug_coffee/SubUSDs/textures/mug_coffee_diffuse.jpg` provide the handled coffee mug. The mug geometry is scaled uniformly to 85%; its original 0.099956 kg mass is scaled by the cube of that factor. The scene uses a cylindrical body collision proxy and the handle is included in the visual mesh.
- `Dinnerware set.usdz` provides the plate.
- `Soda_Can.usdz` provides the soda can.

The selected mesh geometry is extracted and converted to OBJ for MuJoCo. The mug texture is retained with the source package; the scene currently applies a flat material color.

## NVIDIA SimReady Assets

The bottled-water asset is from NVIDIA's [SimReady Assets dataset](https://huggingface.co/datasets/nvidia/simready-assets), asset `food_beverage_bottledWater_b01`, licensed CC-BY-4.0. Its USD payloads, textures, package metadata, and license are retained under `nvidia_bottled_water/`. The mesh is exported from the asset's MuJoCo variant; its bottle and cap geometry is represented by one visual mesh and a cylindrical MuJoCo collision proxy. The proxy uses the combined bottle and cap mass from the SimReady USD metadata.
