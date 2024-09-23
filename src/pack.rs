use std::path::Path;

use hashbrown::HashMap;
use rayon::prelude::*;

use crate::disjoint_set::DisjointSet;
use crate::export::AtlasExporter;
use crate::place::{PlacedPolygonUVCoords, PlacedTextureGeometry, TexturePlacer};
use crate::texture::cache::TextureCache;
use crate::texture::{ChildTexture, PolygonMappedTexture, ToplevelTexture};
pub type Atlas = Vec<PlacedTextureGeometry>;

pub struct AtlasPacker {
    // texture id -> texture
    textures: HashMap<String, PolygonMappedTexture>,
}

impl Default for AtlasPacker {
    fn default() -> Self {
        Self {
            textures: HashMap::new(),
        }
    }
}

#[derive(Clone)]
pub(super) struct Cluster {
    pub toplevel_texture: ToplevelTexture,
    // (texture id, child texture)
    pub children: Vec<(String, ChildTexture)>,
}

impl AtlasPacker {
    pub fn add_texture(&mut self, texture_id: String, texture: PolygonMappedTexture) {
        self.textures.insert(texture_id, texture);
    }

    fn create_cluster(&self) -> HashMap<String, Cluster> {
        let texture_ids: Vec<String> = self.textures.keys().cloned().collect();

        let disjoint_set = {
            let mut disjoint_set = DisjointSet::new(texture_ids.len());

            for i in 0..texture_ids.len() {
                for j in (i + 1)..texture_ids.len() {
                    let texture_i = self.textures.get(&texture_ids[i]).unwrap();
                    let texture_j = self.textures.get(&texture_ids[j]).unwrap();

                    if texture_i.bbox_overlaps(texture_j) {
                        disjoint_set.unite(i, j);
                    }
                }
            }
            disjoint_set.compress();
            disjoint_set
        };

        // cluster id -> texture ids
        let mut cluster_id_map: HashMap<String, Vec<String>> = HashMap::new();
        for i in 0..texture_ids.len() {
            let cluster_id = disjoint_set.root(i).to_string();
            let texture_id = texture_ids[i].clone();
            cluster_id_map
                .entry(cluster_id.to_string())
                .or_insert_with(Vec::new)
                .push(texture_id);
        }

        // create toplevel textures
        let cluster_map = cluster_id_map
            .iter()
            .filter_map(|(cluster_id, texture_ids)| {
                let toplevel_texture = texture_ids
                    .iter()
                    .fold(None, |acc: Option<ToplevelTexture>, texture_id| {
                        let texture = self.textures.get(texture_id).unwrap();
                        match acc {
                            Some(toplevel_texture) => toplevel_texture.expand(texture),
                            None => Some(ToplevelTexture::new(texture)),
                        }
                    })
                    .unwrap();

                let children = texture_ids
                    .iter()
                    .map(|texture_id| {
                        let texture = self.textures.get(texture_id).unwrap();
                        (texture_id.clone(), toplevel_texture.get_child(texture))
                    })
                    .collect::<Vec<(String, ChildTexture)>>();

                Some((
                    cluster_id.clone(),
                    Cluster {
                        toplevel_texture,
                        children,
                    },
                ))
            })
            .collect::<HashMap<String, Cluster>>();

        cluster_map
    }

    pub fn pack<P: TexturePlacer>(self, mut placer: P) -> PackedAtlasProvider {
        let mut current_atlas: Atlas = Vec::new();
        let mut atlases: HashMap<String, Atlas> = HashMap::new();

        let clusters = self.create_cluster();
        let mut texture_info_map: HashMap<String, PlacedPolygonUVCoords> = HashMap::new();
        for (cluster_id, cluster) in clusters.iter() {
            if !placer.can_place(&cluster.toplevel_texture) {
                let current_atlas_id = atlases.len();
                atlases.insert(current_atlas_id.to_string(), current_atlas.clone());
                current_atlas.clear();
                placer.reset_param();
            }

            let current_atlas_id = atlases.len().to_string();

            let (toplevel_texture_info, children_texture_infos) = placer.place_texture(
                cluster.toplevel_texture.clone(),
                cluster.children.clone(),
                cluster_id.clone(),
                current_atlas_id,
            );

            current_atlas.push(toplevel_texture_info.clone());

            for (child_texture_info, child_texture_id) in children_texture_infos
                .iter()
                .zip(cluster.children.iter().map(|(id, _)| id))
            {
                if let Some(child_texture_info) = child_texture_info {
                    texture_info_map.insert(child_texture_id.clone(), child_texture_info.clone());
                }
            }
        }

        // treat the last atlas
        if !current_atlas.is_empty() {
            let current_atlas_id = atlases.len();

            atlases.insert(current_atlas_id.to_string(), current_atlas.clone());
            current_atlas.clear();
        }

        PackedAtlasProvider {
            clusters,
            atlases,
            texture_info_map,
        }
    }
}

pub struct PackedAtlasProvider {
    // atlas id -> atlas
    atlases: HashMap<String, Atlas>,
    // cluster id -> cluster
    clusters: HashMap<String, Cluster>,
    // texture id -> placed texture info
    texture_info_map: HashMap<String, PlacedPolygonUVCoords>,
}

impl PackedAtlasProvider {
    pub fn export<E: AtlasExporter>(
        &self,
        exporter: E,
        output_dir: &Path,
        texture_cache: &TextureCache,
        width: u32,
        height: u32,
    ) {
        self.atlases.par_iter().for_each(|(id, atlas)| {
            let output_path = output_dir.join(id);
            exporter.export(
                atlas,
                &self
                    .clusters
                    .iter()
                    .map(|(id, cluster)| (id.clone(), cluster.toplevel_texture.clone()))
                    .collect::<HashMap<String, ToplevelTexture>>(),
                &output_path,
                texture_cache,
                width,
                height,
            );
        });
    }

    pub fn get_texture_info(&self, id: &str) -> Option<&PlacedPolygonUVCoords> {
        self.texture_info_map.get(id)
    }
}
